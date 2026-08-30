#include "llama-moe-residency.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

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

int llama_moe_dup_fd(int fd) {
#if defined(_WIN32)
    return _dup(fd);
#else
    return dup(fd);
#endif
}

void llama_moe_close_fd(int fd) {
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
            llama_moe_close_fd(f.fd);
            f.fd = -1;
        }
    }
}

llama_moe_status llama_moe_reader::add_fd(int fd, uint64_t size, size_t * idx) {
    if (fd < 0 || size == 0) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    // keep our own descriptor so the reader survives the loader closing its handle
    const int dup_fd = llama_moe_dup_fd(fd);
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

//
// llama_moe_residency
//

// Can experts be read straight into this buffer, or do they have to be staged and copied in?
//
// ggml_backend_buffer_is_host() is the obvious answer and is right for CPU, but it is deliberately false for
// every Metal buffer type even when the memory is genuinely unified and host-addressable. Taking it at face
// value there costs a full second copy of every expert byte, which on unified memory is the dominant cost.
//
// So ask the backend first, through a named entry point, and fall back to the generic answer when a backend
// does not provide one. Looking it up by name keeps this file free of any backend's headers.
static bool llama_moe_buffer_is_host_writable(ggml_backend_buffer_t buf) {
    if (buf == nullptr) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    ggml_backend_dev_t         dev  = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    ggml_backend_reg_t         reg  = dev  ? ggml_backend_dev_backend_reg(dev)  : nullptr;

    if (reg != nullptr) {
        using is_host_writable_t = bool (*)(ggml_backend_buffer_t);

        auto fn = (is_host_writable_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_is_host_writable");

        if (fn != nullptr) {
            return fn(buf);
        }
    }

    return ggml_backend_buffer_is_host(buf);
}

void llama_moe_residency::latch(llama_moe_status status, std::string detail) {
    if (status == LLAMA_MOE_STATUS_OK) {
        return;
    }

    int expected = (int) LLAMA_MOE_STATUS_OK;

    // keep the first failure; later ones are usually consequences of it
    if (first_err.compare_exchange_strong(expected, (int) status, std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(err_mtx);
        err_detail = std::move(detail);
    }
}

llama_moe_status llama_moe_residency::init(
        const llama_moe_params & params,
        const std::unordered_map<std::string, llama_moe_tensor_info> & paged) {
    if (params.n_slots <= 0 || paged.empty()) {
        return LLAMA_MOE_STATUS_OK; // paging is off, or the model had nothing to page
    }

    n_slots    = params.n_slots;
    paged_info = paged;

    int32_t n_layer_max = 0;
    for (const auto & [_, info] : paged) {
        n_layer_max = std::max(n_layer_max, info.il + 1);
    }

    layers.resize((size_t) n_layer_max);

    // sized once so the pointers handed to graph nodes stay valid for the life of the manager
    resolve_ctxs.resize((size_t) n_layer_max);
    for (int32_t il = 0; il < n_layer_max; il++) {
        resolve_ctxs[(size_t) il] = { this, il };
    }

    // one residency map per layer, sized by that layer's expert count
    for (const auto & [name, info] : paged) {
        auto & layer = layers[(size_t) info.il];

        if (layer.cache.n_slots() == 0) {
            const int32_t n_slot = std::min(n_slots, info.n_expert);

            const llama_moe_status status = layer.cache.init(info.il, info.n_expert, n_slot);
            if (status != LLAMA_MOE_STATUS_OK) {
                return status;
            }
        }

        if (layer.cache.n_expert() != info.n_expert) {
            // every pool of a layer is indexed by the same slot, so they must agree on the expert count
            return LLAMA_MOE_STATUS_INVALID_CONFIG;
        }
    }

    const int32_t n_threads = params.n_read_threads > 0 ? params.n_read_threads : 4;

    return reader.start_workers(n_threads);
}

llama_moe_status llama_moe_residency::add_file(int fd, uint64_t size, size_t * idx) {
    return reader.add_fd(fd, size, idx);
}

bool llama_moe_residency::is_paged_layer(int32_t il) const {
    return il >= 0 && (size_t) il < layers.size() && layers[(size_t) il].cache.n_slots() > 0;
}

ggml_tensor * llama_moe_residency::bind_pool(int32_t il, const ggml_tensor * orig, ggml_backend_buffer_type_t buft) {
    if (!is_paged_layer(il) || orig == nullptr || buft == nullptr) {
        return nullptr;
    }

    auto & layer = layers[(size_t) il];

    const std::string name = ggml_get_name(orig);

    // graphs are rebuilt constantly; a pool is created once and reused
    const auto it = layer.by_name.find(name);
    if (it != layer.by_name.end()) {
        return layer.pools[it->second].tensor;
    }

    llama_moe_pool pool;

    {
        ggml_init_params ctx_params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        pool.ctx.reset(ggml_init(ctx_params));
        if (!pool.ctx) {
            latch(LLAMA_MOE_STATUS_BACKEND_ERROR, "failed to create a context for pool " + name);
            return nullptr;
        }
    }

    const int32_t n_slot = layer.cache.n_slots();

    pool.tensor = ggml_new_tensor_3d(pool.ctx.get(), orig->type, orig->ne[0], orig->ne[1], n_slot);
    if (pool.tensor == nullptr) {
        latch(LLAMA_MOE_STATUS_BACKEND_ERROR, "failed to create pool tensor for " + name);
        return nullptr;
    }

    ggml_set_name(pool.tensor, (name + "#pool").c_str());

    pool.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pool.ctx.get(), buft));
    if (!pool.buf) {
        latch(LLAMA_MOE_STATUS_BACKEND_ERROR, "failed to allocate a slot pool for " + name);
        return nullptr;
    }

    ggml_backend_buffer_set_usage(pool.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Zero the pool so an expert's unused tail never contains stale bytes. CUDA MMQ in particular reads a
    // little past the end of the last expert and would otherwise see NaNs there.
    ggml_backend_buffer_clear(pool.buf.get(), 0);

    const auto it_info = paged_info.find(name);
    if (it_info == paged_info.end()) {
        latch(LLAMA_MOE_STATUS_INVALID_CONFIG, "no disk location recorded for paged tensor " + name);
        return nullptr;
    }

    pool.info        = it_info->second;
    pool.slot_stride = pool.tensor->nb[2];  // distance between slots in the pool
    pool.read_size   = pool.info.stride;    // bytes one expert occupies on disk

    pool.host_writable = llama_moe_buffer_is_host_writable(pool.buf.get());

    // The two are separate quantities and treating them as one would silently read the wrong bytes, so
    // require them to agree rather than assuming it.
    if (pool.slot_stride != pool.read_size) {
        latch(LLAMA_MOE_STATUS_INVALID_CONFIG,
                "pool stride does not match the on-disk expert stride for " + name);
        return nullptr;
    }

    ggml_tensor * ret = pool.tensor;

    layer.by_name[name] = layer.pools.size();
    layer.pools.push_back(std::move(pool));

    return ret;
}

llama_moe_status llama_moe_residency::resolve(int32_t il, const int32_t * ids, int32_t n, int32_t * out_slots) {
    if (!is_paged_layer(il)) {
        return LLAMA_MOE_STATUS_INVALID_CONFIG;
    }

    auto & layer = layers[(size_t) il];

    // decide which experts go where; no I/O yet
    const llama_moe_status status = layer.cache.resolve(ids, n, out_slots, fills);
    if (status != LLAMA_MOE_STATUS_OK) {
        // slot 0 always exists, so the matmul reads in-bounds nonsense instead of running off the pool
        std::fill(out_slots, out_slots + n, 0);

        // say what was actually asked for - the slot count needed follows from the routed id count
        latch(status, "layer " + std::to_string(il) + ": " + llama_moe_status_str(status) +
                " (" + std::to_string(n) + " routed ids, " + std::to_string(layer.cache.n_slots()) +
                " slots, " + std::to_string(layer.cache.n_expert()) + " experts)");
        return status;
    }

    if (fills.empty()) {
        return LLAMA_MOE_STATUS_OK; // everything the ubatch needs is already resident
    }

    // every pool of the layer evicts in lockstep, so one admission means one read per pool
    reqs.clear();
    staged.clear();
    reqs.reserve(fills.size() * layer.pools.size());

    // size the staging area for the pools that cannot be written directly
    size_t staging_need = 0;
    for (const auto & pool : layer.pools) {
        if (!pool.host_writable) {
            staging_need += fills.size() * pool.read_size;
        }
    }

    if (staging_need > 0) {
        ggml_backend_buffer_type_t buft = nullptr;
        for (const auto & pool : layer.pools) {
            if (!pool.host_writable && pool.buf) {
                buft = ggml_backend_buffer_get_type(pool.buf.get());
                break;
            }
        }

        const llama_moe_status st = ensure_staging(staging_need, buft);
        if (st != LLAMA_MOE_STATUS_OK) {
            layer.cache.invalidate();
            std::fill(out_slots, out_slots + n, 0);
            latch(st, "failed to allocate expert staging memory");
            return st;
        }
    }

    size_t staging_off = 0;

    for (const auto & pool : layer.pools) {
        uint8_t * base = (uint8_t *) pool.tensor->data;

        if (base == nullptr) {
            latch(LLAMA_MOE_STATUS_BACKEND_ERROR, "slot pool has no storage");
            return LLAMA_MOE_STATUS_BACKEND_ERROR;
        }

        for (const auto & fill : fills) {
            const size_t slot_off = (size_t) fill.slot * pool.slot_stride;

            void * dst;

            if (pool.host_writable) {
                dst = base + slot_off;
            } else {
                dst = staging + staging_off;
                staged.push_back({ &pool, staging_off, slot_off });
                staging_off += pool.read_size;
            }

            reqs.push_back({
                /*.file_idx =*/ pool.info.file_idx,
                /*.offset   =*/ pool.info.offset + (uint64_t) fill.expert * pool.info.stride,
                /*.dst      =*/ dst,
                /*.size     =*/ pool.read_size,
            });
        }
    }

    const llama_moe_status io = reader.read_many(reqs.data(), reqs.size());

    if (io != LLAMA_MOE_STATUS_OK) {
        // the slots that were being filled hold undefined bytes now, so drop residency for the layer
        layer.cache.invalidate();
        std::fill(out_slots, out_slots + n, 0);
        latch(io, "layer " + std::to_string(il) + ": " + llama_moe_status_str(io));
        return io;
    }

    // hand the staged experts to the backend now that they are all read
    for (const auto & copy : staged) {
        ggml_backend_tensor_set(copy.pool->tensor, staging + copy.staging_off, copy.slot_off, copy.pool->read_size);
    }

    return LLAMA_MOE_STATUS_OK;
}

llama_moe_status llama_moe_residency::ensure_staging(size_t n_bytes, ggml_backend_buffer_type_t buft) {
    if (n_bytes <= staging_cap && staging != nullptr) {
        return LLAMA_MOE_STATUS_OK;
    }

    // grow generously so a steady state is reached quickly and no allocation happens per token
    const size_t want = n_bytes + n_bytes/2;

    // Pinned host memory where the device offers it, so the host-to-device copies are not bounced through
    // pageable memory. Not every backend has one, hence the plain fallback.
    ggml_backend_dev_t dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;

    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;

    if (host_buft != nullptr) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(host_buft, want);

        if (buf != nullptr) {
            staging_buf.reset(buf);
            staging_vec.clear();
            staging_vec.shrink_to_fit();
            staging     = (uint8_t *) ggml_backend_buffer_get_base(buf);
            staging_cap = want;

            return LLAMA_MOE_STATUS_OK;
        }
        // fall through to ordinary memory if pinning failed
    }

    staging_buf.reset();
    staging_vec.resize(want);
    staging     = staging_vec.data();
    staging_cap = want;

    return LLAMA_MOE_STATUS_OK;
}

llama_moe_resolve_ctx * llama_moe_residency::resolve_ctx(int32_t il) {
    if (il < 0 || (size_t) il >= resolve_ctxs.size()) {
        return nullptr;
    }

    return &resolve_ctxs[(size_t) il];
}

void llama_moe_residency::shutdown() {
    reader.shutdown();
}

llama_moe_layer_stats llama_moe_residency::layer_stats(int32_t il) const {
    if (!is_paged_layer(il)) {
        return {};
    }

    return layers[(size_t) il].cache.stats();
}

llama_moe_layer_stats llama_moe_residency::total_stats() const {
    llama_moe_layer_stats out;

    for (const auto & layer : layers) {
        const auto & st = layer.cache.stats();

        out.n_resolve += st.n_resolve;
        out.n_lookup  += st.n_lookup;
        out.n_hit     += st.n_hit;
        out.n_miss    += st.n_miss;
        out.n_evict   += st.n_evict;
    }

    return out;
}

void llama_moe_residency::reset_stats() {
    for (auto & layer : layers) {
        layer.cache.reset_stats();
    }

    reader.reset_stats();
}
