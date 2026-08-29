#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Bounded per-layer residency of MoE expert weights.
//
// A MoE layer holds n_expert expert weight matrices, but only n_expert_used of them are needed per token.
// Instead of keeping all of them resident, this keeps a bounded pool of n_slots of them and pages the rest
// in from the model file on demand, evicting on an LRU basis.
//
// The routed expert ids are only known once the router has run, so the caller resolves them to slot indices
// per ubatch. resolve() is pure bookkeeping - it decides what must be paged in but performs no I/O, which
// keeps the policy testable in isolation and independent of any backend.

enum llama_moe_status {
    LLAMA_MOE_STATUS_OK = 0,
    LLAMA_MOE_STATUS_INVALID_CONFIG,   // nonsensical n_expert/n_slots combination
    LLAMA_MOE_STATUS_INVALID_EXPERT,   // routed expert id outside [0, n_expert)
    LLAMA_MOE_STATUS_SLOTS_EXHAUSTED,  // one ubatch routed to more distinct experts than there are slots
    LLAMA_MOE_STATUS_IO_ERROR,         // read failed
    LLAMA_MOE_STATUS_SHORT_READ,       // read returned fewer bytes than the expert occupies
    LLAMA_MOE_STATUS_BACKEND_ERROR,    // allocation or transfer failed in a backend adapter
    LLAMA_MOE_STATUS_CANCELLED,        // shutdown requested while work was outstanding
};

const char * llama_moe_status_str(llama_moe_status status);

// an expert that must be read into a slot before the slot may be used
struct llama_moe_fill {
    int32_t expert;
    int32_t slot;
};

struct llama_moe_layer_stats {
    uint64_t n_resolve = 0;  // resolve() calls
    uint64_t n_lookup  = 0;  // routed ids examined
    uint64_t n_hit     = 0;  // ids that were already resident
    uint64_t n_miss    = 0;  // ids that had to be paged in
    uint64_t n_evict   = 0;  // resident experts displaced to make room
};

// The residency map for a single MoE layer.
//
// All pools of one layer (gate/up/down) share this map so that they evict in lockstep - a slot index means
// the same expert in every pool of the layer, which is what lets one remapped id array drive all of them.
struct llama_moe_layer_cache {
    // n_expert: experts in the layer, n_slots: how many stay resident
    llama_moe_status init(int32_t il, int32_t n_expert, int32_t n_slots);

    // Map one ubatch worth of routed expert ids onto slot indices.
    //
    // ids/out_slots hold n entries; out_slots receives the slot each id resolved to. Experts that were not
    // already resident are appended to fills and must be read in before the slots are used. A repeated id
    // within one call resolves to the same slot and is only filled once.
    //
    // Slots touched by this call are pinned for its duration, so an expert admitted early in an ubatch
    // cannot be evicted by a later id in the same ubatch. That makes n_slots < distinct experts in the
    // ubatch unsatisfiable rather than silently wrong, and it is reported as LLAMA_MOE_STATUS_SLOTS_EXHAUSTED.
    llama_moe_status resolve(const int32_t * ids, int32_t n, int32_t * out_slots, std::vector<llama_moe_fill> & fills);

    // drop all residency, e.g. after an I/O failure leaves slot contents undefined
    void invalidate();

    int32_t layer   () const { return il;      }
    int32_t n_expert() const { return n_exp;   }
    int32_t n_slots () const { return n_slot;  }

    // slot holding expert e, or -1 if not resident (for tests and diagnostics)
    int32_t slot_of(int32_t expert) const;
    // expert held in slot s, or -1 if empty
    int32_t expert_in(int32_t slot) const;

    const llama_moe_layer_stats & stats() const { return st; }
    void reset_stats() { st = {}; }

private:
    // least recently used slot that is not pinned by the current resolve(), or -1 if all are pinned
    int32_t evict_victim() const;

    int32_t il    = -1;
    int32_t n_exp = 0;
    int32_t n_slot = 0;

    std::vector<int32_t> expert_to_slot;  // n_exp entries, -1 when not resident
    std::vector<int32_t> slot_to_expert;  // n_slot entries, -1 when empty
    std::vector<uint64_t> lru_clock;      // n_slot entries, last touch
    std::vector<bool>     pinned;         // n_slot entries, in use by the current resolve()

    uint64_t clock = 0;

    llama_moe_layer_stats st;
};

// Smallest pool that can serve any ubatch: a ubatch routes at most n_expert_used experts per token, and
// never more distinct experts than the layer has.
int32_t llama_moe_min_slots(int32_t n_expert, int32_t n_expert_used, int32_t n_ubatch);

//
// reading expert weights from the model file
//

struct llama_moe_io_stats {
    uint64_t n_read    = 0;  // completed reads
    uint64_t n_bytes   = 0;  // bytes delivered
    uint64_t t_read_us = 0;  // wall time spent inside read_many(), including waiting on workers
};

// one expert's worth of bytes to move from the model file into a destination the caller owns
struct llama_moe_read_req {
    size_t   file_idx;
    uint64_t offset;
    void *   dst;
    size_t   size;
};

// Positional reader over the model file(s) holding expert weights.
//
// Reads are positional so that many can be in flight on one descriptor without seek races, and every read
// is bounds-checked against the file length before it is issued. Interrupted and partial reads are retried;
// a genuine short read is reported rather than leaving a slot half-written.
struct llama_moe_reader {
    llama_moe_reader() = default;
    ~llama_moe_reader();

    llama_moe_reader(const llama_moe_reader &) = delete;
    llama_moe_reader & operator=(const llama_moe_reader &) = delete;

    // Duplicate fd and take ownership of the duplicate, so the reader keeps working after the loader has
    // closed its own handle. size is the file length, used to bounds-check every subsequent read.
    llama_moe_status add_fd(int fd, uint64_t size, size_t * idx);

    // open by path; for tests and tools
    llama_moe_status add_path(const char * path, size_t * idx);

    // Read one request. Safe to call concurrently.
    llama_moe_status read(const llama_moe_read_req & req) const;

    // Read a batch, spreading it over the worker threads. Returns the first error encountered; on error
    // some requests may not have run, so the caller must treat every destination as undefined.
    //
    // One batch at a time: this may not be called concurrently with itself. The calling thread takes part
    // in the batch, so it completes even if no workers were started.
    llama_moe_status read_many(const llama_moe_read_req * reqs, size_t n);

    // Start n_threads workers. 0 or 1 keeps everything on the calling thread.
    llama_moe_status start_workers(int32_t n_threads);

    // Ask outstanding and future work to stop, then join. Safe to call more than once.
    void shutdown();

    size_t   n_files()             const { return files.size(); }
    uint64_t file_size(size_t idx) const;
    int32_t  n_threads()           const { return n_worker; }

    llama_moe_io_stats stats() const;
    void reset_stats();

private:
    void worker_loop();
    // claim and service requests until the batch is exhausted; run by the workers and the calling thread
    void drain(const llama_moe_read_req * reqs, size_t n);

    struct file_entry {
        int      fd   = -1;
        uint64_t size = 0;
    };

    std::vector<file_entry> files;

    // worker pool; batches are handed out by atomically claiming indices into the current request array
    std::vector<std::thread> workers;
    int32_t                  n_worker = 0;

    mutable std::mutex      mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;

    const llama_moe_read_req * cur_reqs = nullptr;
    size_t                     cur_n    = 0;
    uint64_t                   gen      = 0;  // bumped per batch so workers can tell batches apart

    std::atomic<size_t>   next_idx {0};
    std::atomic<size_t>   n_done   {0};
    std::atomic<int>      first_err{(int) LLAMA_MOE_STATUS_OK};
    std::atomic<bool>     stopping {false};

    // read() is const but accounts for what it moved, so these are mutable and atomic
    mutable std::atomic<uint64_t> st_n_read {0};
    mutable std::atomic<uint64_t> st_n_bytes{0};
    std::atomic<uint64_t>         st_t_read_us{0};
};
