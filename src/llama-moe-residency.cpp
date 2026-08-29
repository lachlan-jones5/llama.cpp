#include "llama-moe-residency.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <io.h>
#else
#   include <cerrno>
#   include <unistd.h>
#endif

const char * llama_moe_status_str(llama_moe_status status) {
    switch (status) {
        case LLAMA_MOE_STATUS_OK:               return "ok";
        case LLAMA_MOE_STATUS_INVALID_CONFIG:   return "invalid expert residency configuration";
        case LLAMA_MOE_STATUS_INVALID_EXPERT:   return "routed expert id out of range";
        case LLAMA_MOE_STATUS_SLOTS_EXHAUSTED:  return "ubatch routed to more distinct experts than there are slots";
        case LLAMA_MOE_STATUS_IO_ERROR:         return "failed to read expert weights";
        case LLAMA_MOE_STATUS_SHORT_READ:       return "short read while reading expert weights";
        case LLAMA_MOE_STATUS_BACKEND_ERROR:    return "backend rejected an expert residency operation";
        case LLAMA_MOE_STATUS_CANCELLED:        return "expert residency cancelled";
    }

    return "unknown expert residency status";
}

int32_t llama_moe_min_slots(int32_t n_expert, int32_t n_expert_used, int32_t n_ubatch) {
    if (n_expert <= 0 || n_expert_used <= 0 || n_ubatch <= 0) {
        return 0;
    }

    // widen: n_expert_used * n_ubatch overflows int32 for large batches
    const int64_t needed = (int64_t) n_expert_used * (int64_t) n_ubatch;

    return (int32_t) std::min<int64_t>(needed, n_expert);
}

llama_moe_status llama_moe_layer_cache::init(int32_t il, int32_t n_expert, int32_t n_slots) {
    if (n_expert <= 0 || n_slots <= 0 || n_slots > n_expert) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    this->il    = il;
    this->n_exp = n_expert;
    this->n_slot = n_slots;

    expert_to_slot.assign(n_expert, -1);
    slot_to_expert.assign(n_slots,  -1);
    lru_clock     .assign(n_slots,   0);
    pinned        .assign(n_slots, false);

    clock = 0;
    st    = {};

    return LLAMA_MOE_STATUS_OK;
}

void llama_moe_layer_cache::invalidate() {
    std::fill(expert_to_slot.begin(), expert_to_slot.end(), -1);
    std::fill(slot_to_expert.begin(), slot_to_expert.end(), -1);
    std::fill(lru_clock.begin(),      lru_clock.end(),       0);
    std::fill(pinned.begin(),         pinned.end(),      false);

    clock = 0;
}

int32_t llama_moe_layer_cache::slot_of(int32_t expert) const {
    if (expert < 0 || expert >= n_exp) {
        return -1;
    }

    return expert_to_slot[expert];
}

int32_t llama_moe_layer_cache::expert_in(int32_t slot) const {
    if (slot < 0 || slot >= n_slot) {
        return -1;
    }

    return slot_to_expert[slot];
}

int32_t llama_moe_layer_cache::evict_victim() const {
    int32_t  best       = -1;
    uint64_t best_clock = 0;

    for (int32_t s = 0; s < n_slot; s++) {
        if (pinned[s]) {
            continue;
        }

        // empty slots carry clock 0 and so are always preferred over resident ones
        if (best < 0 || lru_clock[s] < best_clock) {
            best       = s;
            best_clock = lru_clock[s];
        }
    }

    return best;
}

llama_moe_status llama_moe_layer_cache::resolve(
        const int32_t * ids,
              int32_t   n,
              int32_t * out_slots,
        std::vector<llama_moe_fill> & fills) {
    fills.clear();

    if (n_slot <= 0) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    if (n < 0 || (n > 0 && (ids == nullptr || out_slots == nullptr))) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    // nothing from a previous ubatch stays pinned
    std::fill(pinned.begin(), pinned.end(), false);

    st.n_resolve++;

    for (int32_t i = 0; i < n; i++) {
        const int32_t e = ids[i];

        if (e < 0 || e >= n_exp) {
            // the map is left coherent, but any slot admitted by this call holds no data yet
            invalidate();
            fills.clear();
            return LLAMA_MOE_STATUS_INVALID_EXPERT;
        }

        st.n_lookup++;

        int32_t s = expert_to_slot[e];

        if (s >= 0) {
            st.n_hit++;
        } else {
            s = evict_victim();

            if (s < 0) {
                // every slot is pinned by this same ubatch, so no eviction can help
                invalidate();
                fills.clear();
                return LLAMA_MOE_STATUS_SLOTS_EXHAUSTED;
            }

            const int32_t prev = slot_to_expert[s];

            if (prev >= 0) {
                expert_to_slot[prev] = -1;
                st.n_evict++;
            }

            slot_to_expert[s] = e;
            expert_to_slot[e] = s;

            st.n_miss++;

            fills.push_back({ e, s });
        }

        lru_clock[s] = ++clock;
        pinned[s]    = true;

        out_slots[i] = s;
    }

    return LLAMA_MOE_STATUS_OK;
}

//
// llama_moe_reader
//

// Read exactly size bytes at offset, retrying interrupted and partial reads.
// Returns the number of bytes delivered, or -1 if the underlying read failed.
static int64_t llama_moe_pread(int fd, void * dst, size_t size, uint64_t offset) {
    uint8_t * p   = (uint8_t *) dst;
    size_t    got = 0;

    while (got < size) {
#if defined(_WIN32)
        const HANDLE h = (HANDLE) _get_osfhandle(fd);
        if (h == INVALID_HANDLE_VALUE) {
            return -1;
        }

        const uint64_t off = offset + got;

        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (off >> 32);

        // cap so the cast to DWORD is always well defined
        const DWORD want = (DWORD) std::min<size_t>(size - got, 1u << 30);

        DWORD n = 0;
        if (!ReadFile(h, p + got, want, &n, &ov)) {
            if (GetLastError() == ERROR_HANDLE_EOF) {
                break;
            }
            return -1;
        }
        if (n == 0) {
            break; // end of file
        }
        got += (size_t) n;
#else
        const ssize_t n = pread(fd, p + got, size - got, (off_t) (offset + got));
        if (n < 0) {
            if (errno == EINTR) {
                continue; // interrupted before transferring anything - retry
            }
            return -1;
        }
        if (n == 0) {
            break; // end of file
        }
        got += (size_t) n;
#endif
    }

    return (int64_t) got;
}

static int llama_moe_dup(int fd) {
#if defined(_WIN32)
    return _dup(fd);
#else
    return dup(fd);
#endif
}

static void llama_moe_close(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}

llama_moe_reader::~llama_moe_reader() {
    shutdown();

    for (auto & f : files) {
        if (f.fd >= 0) {
            llama_moe_close(f.fd);
            f.fd = -1;
        }
    }
}

llama_moe_status llama_moe_reader::add_fd(int fd, uint64_t size, size_t * idx) {
    if (fd < 0 || size == 0) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    // keep our own descriptor so the reader survives the loader closing its handle
    const int dup_fd = llama_moe_dup(fd);
    if (dup_fd < 0) {
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    files.push_back({ dup_fd, size });

    if (idx) {
        *idx = files.size() - 1;
    }

    return LLAMA_MOE_STATUS_OK;
}

llama_moe_status llama_moe_reader::add_path(const char * path, size_t * idx) {
    if (path == nullptr) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    const long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    const llama_moe_status status = add_fd(fileno(f), (uint64_t) size, idx);

    fclose(f);

    return status;
}

uint64_t llama_moe_reader::file_size(size_t idx) const {
    if (idx >= files.size()) {
        return 0;
    }

    return files[idx].size;
}

llama_moe_status llama_moe_reader::read(const llama_moe_read_req & req) const {
    if (req.file_idx >= files.size() || req.dst == nullptr) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    if (req.size == 0) {
        return LLAMA_MOE_STATUS_OK;
    }

    const file_entry & f = files[req.file_idx];

    // reject the read rather than discovering the truncation halfway through it
    if (req.offset > f.size || req.size > f.size - req.offset) {
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    const int64_t got = llama_moe_pread(f.fd, req.dst, req.size, req.offset);

    if (got < 0) {
        return LLAMA_MOE_STATUS_IO_ERROR;
    }

    if ((size_t) got != req.size) {
        return LLAMA_MOE_STATUS_SHORT_READ;
    }

    st_n_read .fetch_add(1,        std::memory_order_relaxed);
    st_n_bytes.fetch_add(req.size, std::memory_order_relaxed);

    return LLAMA_MOE_STATUS_OK;
}

void llama_moe_reader::drain(const llama_moe_read_req * reqs, size_t n) {
    for (;;) {
        const size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);

        if (i >= n) {
            break;
        }

        llama_moe_status status;

        if (stopping.load(std::memory_order_relaxed)) {
            status = LLAMA_MOE_STATUS_CANCELLED;
        } else {
            status = read(reqs[i]);
        }

        if (status != LLAMA_MOE_STATUS_OK) {
            int expected = (int) LLAMA_MOE_STATUS_OK;
            first_err.compare_exchange_strong(expected, (int) status, std::memory_order_relaxed);
        }

        if (n_done.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            std::lock_guard<std::mutex> lock(mtx);
            cv_done.notify_all();
        }
    }
}

void llama_moe_reader::worker_loop() {
    uint64_t seen = 0;

    for (;;) {
        const llama_moe_read_req * reqs = nullptr;
        size_t                     n    = 0;

        {
            std::unique_lock<std::mutex> lock(mtx);

            cv_work.wait(lock, [this, seen] { return stopping.load(std::memory_order_relaxed) || gen != seen; });

            if (stopping.load(std::memory_order_relaxed)) {
                return;
            }

            seen = gen;
            reqs = cur_reqs;
            n    = cur_n;
        }

        if (reqs != nullptr) {
            drain(reqs, n);
        }
    }
}

llama_moe_status llama_moe_reader::start_workers(int32_t n_threads) {
    if (!workers.empty()) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    if (stopping.load(std::memory_order_relaxed)) {
        return LLAMA_MOE_STATUS_CANCELLED;
    }

    // 0 or 1 means everything runs on the calling thread
    if (n_threads <= 1) {
        n_worker = n_threads < 0 ? 0 : n_threads;
        return LLAMA_MOE_STATUS_OK;
    }

    n_worker = n_threads;

    workers.reserve((size_t) n_threads);
    for (int32_t i = 0; i < n_threads; i++) {
        workers.emplace_back(&llama_moe_reader::worker_loop, this);
    }

    return LLAMA_MOE_STATUS_OK;
}

llama_moe_status llama_moe_reader::read_many(const llama_moe_read_req * reqs, size_t n) {
    if (n == 0) {
        return LLAMA_MOE_STATUS_OK;
    }

    if (reqs == nullptr) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    if (stopping.load(std::memory_order_relaxed)) {
        return LLAMA_MOE_STATUS_CANCELLED;
    }

    const auto t_start = std::chrono::steady_clock::now();

    llama_moe_status status = LLAMA_MOE_STATUS_OK;

    if (workers.empty() || n == 1) {
        // not worth waking anyone
        for (size_t i = 0; i < n; i++) {
            const llama_moe_status s = read(reqs[i]);
            if (s != LLAMA_MOE_STATUS_OK && status == LLAMA_MOE_STATUS_OK) {
                status = s;
            }
        }
    } else {
        {
            std::lock_guard<std::mutex> lock(mtx);

            cur_reqs = reqs;
            cur_n    = n;

            next_idx .store(0, std::memory_order_relaxed);
            n_done   .store(0, std::memory_order_relaxed);
            first_err.store((int) LLAMA_MOE_STATUS_OK, std::memory_order_relaxed);

            gen++;
        }

        cv_work.notify_all();

        // the caller is a worker too, so the batch still completes even if every worker has exited
        drain(reqs, n);

        {
            std::unique_lock<std::mutex> lock(mtx);

            cv_done.wait(lock, [this] { return n_done.load(std::memory_order_acquire) >= cur_n; });

            cur_reqs = nullptr;
            cur_n    = 0;
        }

        status = (llama_moe_status) first_err.load(std::memory_order_relaxed);
    }

    const auto t_end = std::chrono::steady_clock::now();

    st_t_read_us.fetch_add(
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count(),
        std::memory_order_relaxed);

    return status;
}

void llama_moe_reader::shutdown() {
    if (workers.empty()) {
        stopping.store(true, std::memory_order_relaxed);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        stopping.store(true, std::memory_order_relaxed);
    }

    cv_work.notify_all();

    for (auto & t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    workers.clear();
    n_worker = 0;
}

llama_moe_io_stats llama_moe_reader::stats() const {
    llama_moe_io_stats out;

    out.n_read    = st_n_read   .load(std::memory_order_relaxed);
    out.n_bytes   = st_n_bytes  .load(std::memory_order_relaxed);
    out.t_read_us = st_t_read_us.load(std::memory_order_relaxed);

    return out;
}

void llama_moe_reader::reset_stats() {
    st_n_read   .store(0, std::memory_order_relaxed);
    st_n_bytes  .store(0, std::memory_order_relaxed);
    st_t_read_us.store(0, std::memory_order_relaxed);
}
