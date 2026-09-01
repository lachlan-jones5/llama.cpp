// Unit tests for the backend-independent MoE expert residency state machine.
//
// This exercises policy only - no model, no backend, no I/O.

#include "../src/llama-moe-residency.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                   \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                     \
            n_fail++;                                                                    \
        }                                                                                \
    } while (0)

// resolve a ubatch and hand back the slots it mapped to
static llama_moe_status run(
        llama_moe_layer_cache & cache,
        const std::vector<int32_t> & ids,
        std::vector<int32_t> & slots,
        std::vector<llama_moe_fill> & fills) {
    slots.assign(ids.size(), -1);

    return cache.resolve(ids.data(), (int32_t) ids.size(), slots.data(), fills);
}

static void test_min_slots() {
    printf("test_min_slots\n");

    // bounded by the number of distinct experts that can be routed to
    CHECK(llama_moe_min_slots(128, 8, 1)  == 8);
    CHECK(llama_moe_min_slots(128, 8, 4)  == 32);
    // ...but never more than the layer has
    CHECK(llama_moe_min_slots(128, 8, 64) == 128);
    CHECK(llama_moe_min_slots(8,   8, 1)  == 8);

    // must not overflow when widening n_expert_used * n_ubatch
    CHECK(llama_moe_min_slots(64, 1 << 20, 1 << 20) == 64);

    CHECK(llama_moe_min_slots(0, 8, 1) == 0);
    CHECK(llama_moe_min_slots(8, 0, 1) == 0);
}

static void test_init_validation() {
    printf("test_init_validation\n");

    llama_moe_layer_cache cache;

    CHECK(cache.init(0,  0,  4) == LLAMA_MOE_STATUS_INVALID_CONFIG);  // no experts
    CHECK(cache.init(0,  8,  0) == LLAMA_MOE_STATUS_INVALID_CONFIG);  // no slots
    CHECK(cache.init(0,  8, -1) == LLAMA_MOE_STATUS_INVALID_CONFIG);
    CHECK(cache.init(0,  8,  9) == LLAMA_MOE_STATUS_INVALID_CONFIG);  // more slots than experts

    CHECK(cache.init(3,  8,  4) == LLAMA_MOE_STATUS_OK);
    CHECK(cache.layer()    == 3);
    CHECK(cache.n_expert() == 8);
    CHECK(cache.n_slots()  == 4);

    // a fresh cache holds nothing
    for (int32_t e = 0; e < 8; e++) {
        CHECK(cache.slot_of(e) == -1);
    }
    for (int32_t s = 0; s < 4; s++) {
        CHECK(cache.expert_in(s) == -1);
    }

    // out-of-range queries are answered, not crashed
    CHECK(cache.slot_of(-1)  == -1);
    CHECK(cache.slot_of(999) == -1);
    CHECK(cache.expert_in(-1)  == -1);
    CHECK(cache.expert_in(999) == -1);
}

static void test_miss_then_hit() {
    printf("test_miss_then_hit\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 4) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // cold: every id is a miss and must be filled
    CHECK(run(cache, {0, 1}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.size() == 2);
    CHECK(cache.stats().n_miss == 2);
    CHECK(cache.stats().n_hit  == 0);
    CHECK(cache.stats().n_evict == 0);

    // the reported fills agree with the map
    for (const auto & f : fills) {
        CHECK(cache.slot_of(f.expert)  == f.slot);
        CHECK(cache.expert_in(f.slot)  == f.expert);
    }

    const int32_t slot0 = cache.slot_of(0);
    const int32_t slot1 = cache.slot_of(1);
    CHECK(slot0 != slot1);

    // warm: same experts, no fills, same slots
    CHECK(run(cache, {0, 1}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.empty());
    CHECK(cache.stats().n_hit  == 2);
    CHECK(cache.stats().n_miss == 2);
    CHECK(slots[0] == slot0);
    CHECK(slots[1] == slot1);
}

static void test_repeated_id_fills_once() {
    printf("test_repeated_id_fills_once\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 4) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // the same expert routed by several tokens in one ubatch occupies one slot and is read once
    CHECK(run(cache, {5, 5, 5, 5}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.size() == 1);
    CHECK(cache.stats().n_miss == 1);
    CHECK(cache.stats().n_hit  == 3);
    CHECK(slots[0] == slots[1]);
    CHECK(slots[1] == slots[2]);
    CHECK(slots[2] == slots[3]);
}

// The per-layer embedding table has hundreds of millions of rows, so the cache cannot keep an array entry
// per id the way it does for a few hundred experts. Everything about resolving must still behave the same;
// what changes is that use counts follow the slot, so they start over on admission rather than surviving
// eviction.
static void test_large_id_space_stays_sparse() {
    printf("test_large_id_space_stays_sparse\n");

    const int32_t n_ids   = 320000000;  // the real table's order of magnitude
    const int32_t n_slots = 64;

    llama_moe_layer_cache cache;
    CHECK(cache.init(-1, n_ids, n_slots) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // ids from all over the space, including the very last row
    const int32_t last = n_ids - 1;

    CHECK(run(cache, {0, 1234567, last, 1234567}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.size() == 3);          // the repeat is filled once
    CHECK(slots[1] == slots[3]);
    CHECK(cache.stats().n_hit == 1);
    CHECK(cache.slot_of(last) != -1);
    CHECK(cache.slot_of(2) == -1);

    // a second ubatch finds them all resident
    CHECK(run(cache, {0, 1234567, last}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.empty());

    // out-of-range is still rejected on the id, not on the map
    CHECK(run(cache, {n_ids}, slots, fills) == LLAMA_MOE_STATUS_INVALID_EXPERT);

    // Fill every slot, then push one more id in. Frequency still decides: 0 was routed twice above, the
    // filler ids once each, so a filler goes rather than 0.
    CHECK(cache.init(-1, n_ids, 4) == LLAMA_MOE_STATUS_OK);

    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {100000000}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {200000000}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {300000000}, slots, fills) == LLAMA_MOE_STATUS_OK);

    CHECK(run(cache, {123}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(cache.slot_of(0) != -1);
    CHECK(cache.slot_of(123) != -1);
    CHECK(cache.stats().n_evict == 1);

    // invalidate drops residency and leaves the map usable
    cache.invalidate();
    CHECK(cache.slot_of(0) == -1);
    CHECK(run(cache, {42}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(cache.slot_of(42) != -1);
}

static void test_lfu_eviction_order() {
    printf("test_lfu_eviction_order\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 2) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // 0 is routed often, 1 rarely but more recently. This is the case where frequency and recency disagree:
    // LRU would evict 0 because it has not been touched for two ubatches, which is exactly the wrong answer.
    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {1}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {1}, slots, fills) == LLAMA_MOE_STATUS_OK);

    // admitting 2 must displace the rarely routed 1, keeping the popular 0
    CHECK(run(cache, {2}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.size() == 1);
    CHECK(cache.slot_of(1) == -1);
    CHECK(cache.slot_of(0) != -1);
    CHECK(cache.slot_of(2) != -1);
    CHECK(cache.stats().n_evict == 1);
}

static void test_recency_breaks_frequency_ties() {
    printf("test_recency_breaks_frequency_ties\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 2) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // both routed exactly once, so only recency separates them
    CHECK(run(cache, {0}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(run(cache, {1}, slots, fills) == LLAMA_MOE_STATUS_OK);

    // 2 displaces the older of the two
    CHECK(run(cache, {2}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(cache.slot_of(0) == -1);
    CHECK(cache.slot_of(1) != -1);
    CHECK(cache.slot_of(2) != -1);
}

static void test_in_ubatch_pinning() {
    printf("test_in_ubatch_pinning\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 2) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    // two distinct experts exactly fill the pool, and both stay valid for the whole ubatch
    CHECK(run(cache, {3, 4}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(slots[0] != slots[1]);
    CHECK(cache.slot_of(3) != -1);
    CHECK(cache.slot_of(4) != -1);

    // a third distinct expert in the same ubatch cannot be served: evicting either of the first two
    // would corrupt a matmul that still needs them
    CHECK(run(cache, {3, 4, 5}, slots, fills) == LLAMA_MOE_STATUS_SLOTS_EXHAUSTED);

    // the failure must not leave slots claiming to hold data that was never read
    for (int32_t e = 0; e < 8; e++) {
        CHECK(cache.slot_of(e) == -1);
    }
    CHECK(fills.empty());
}

static void test_minimum_slot_boundary() {
    printf("test_minimum_slot_boundary\n");

    const int32_t n_expert      = 64;
    const int32_t n_expert_used = 8;

    // exactly the minimum pool for a single-token ubatch succeeds
    {
        llama_moe_layer_cache cache;
        const int32_t n_slots = llama_moe_min_slots(n_expert, n_expert_used, 1);
        CHECK(n_slots == 8);
        CHECK(cache.init(0, n_expert, n_slots) == LLAMA_MOE_STATUS_OK);

        std::vector<int32_t>        slots;
        std::vector<llama_moe_fill> fills;
        CHECK(run(cache, {0, 1, 2, 3, 4, 5, 6, 7}, slots, fills) == LLAMA_MOE_STATUS_OK);
        CHECK(fills.size() == 8);
    }

    // one slot short of the minimum fails cleanly instead of corrupting
    {
        llama_moe_layer_cache cache;
        CHECK(cache.init(0, n_expert, n_expert_used - 1) == LLAMA_MOE_STATUS_OK);

        std::vector<int32_t>        slots;
        std::vector<llama_moe_fill> fills;
        CHECK(run(cache, {0, 1, 2, 3, 4, 5, 6, 7}, slots, fills) == LLAMA_MOE_STATUS_SLOTS_EXHAUSTED);
    }
}

static void test_invalid_expert_id() {
    printf("test_invalid_expert_id\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 4) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    CHECK(run(cache, {8},  slots, fills) == LLAMA_MOE_STATUS_INVALID_EXPERT);
    CHECK(run(cache, {-1}, slots, fills) == LLAMA_MOE_STATUS_INVALID_EXPERT);

    // a bad id partway through a ubatch must not leave the earlier admissions looking resident
    CHECK(run(cache, {0, 1, 99}, slots, fills) == LLAMA_MOE_STATUS_INVALID_EXPERT);
    for (int32_t e = 0; e < 8; e++) {
        CHECK(cache.slot_of(e) == -1);
    }
    CHECK(fills.empty());
}

static void test_empty_and_bad_arguments() {
    printf("test_empty_and_bad_arguments\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 4) == LLAMA_MOE_STATUS_OK);

    std::vector<llama_moe_fill> fills;

    // an empty ubatch is legal and asks for nothing
    CHECK(cache.resolve(nullptr, 0, nullptr, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.empty());

    int32_t slot = -1;
    int32_t id   = 0;
    CHECK(cache.resolve(nullptr, 1, &slot,   fills) == LLAMA_MOE_STATUS_INVALID_CONFIG);
    CHECK(cache.resolve(&id,     1, nullptr, fills) == LLAMA_MOE_STATUS_INVALID_CONFIG);
    CHECK(cache.resolve(&id,    -1, &slot,   fills) == LLAMA_MOE_STATUS_INVALID_CONFIG);

    // an uninitialised cache refuses work rather than dividing by zero
    llama_moe_layer_cache fresh;
    CHECK(fresh.resolve(&id, 1, &slot, fills) == LLAMA_MOE_STATUS_INVALID_CONFIG);
}

static void test_invalidate() {
    printf("test_invalidate\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 8, 4) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    CHECK(run(cache, {0, 1, 2}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(cache.slot_of(1) != -1);

    // after a failed read the slot contents are undefined, so residency must be dropped
    cache.invalidate();

    for (int32_t e = 0; e < 8; e++) {
        CHECK(cache.slot_of(e) == -1);
    }
    for (int32_t s = 0; s < 4; s++) {
        CHECK(cache.expert_in(s) == -1);
    }

    // and the next ubatch re-reads everything
    CHECK(run(cache, {0, 1, 2}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.size() == 3);
}

static void test_stats_accounting() {
    printf("test_stats_accounting\n");

    llama_moe_layer_cache cache;
    CHECK(cache.init(0, 4, 2) == LLAMA_MOE_STATUS_OK);

    std::vector<int32_t>        slots;
    std::vector<llama_moe_fill> fills;

    CHECK(run(cache, {0, 1}, slots, fills) == LLAMA_MOE_STATUS_OK);  // 2 miss
    CHECK(run(cache, {0, 1}, slots, fills) == LLAMA_MOE_STATUS_OK);  // 2 hit
    CHECK(run(cache, {2, 3}, slots, fills) == LLAMA_MOE_STATUS_OK);  // 2 miss, 2 evict

    const auto & st = cache.stats();
    CHECK(st.n_resolve == 3);
    CHECK(st.n_lookup  == 6);
    CHECK(st.n_hit     == 2);
    CHECK(st.n_miss    == 4);
    CHECK(st.n_evict   == 2);
    CHECK(st.n_hit + st.n_miss == st.n_lookup);

    cache.reset_stats();
    CHECK(cache.stats().n_lookup == 0);
    CHECK(cache.stats().n_hit    == 0);

    // resetting statistics must not disturb residency
    CHECK(run(cache, {2, 3}, slots, fills) == LLAMA_MOE_STATUS_OK);
    CHECK(fills.empty());
}

static void test_status_strings() {
    printf("test_status_strings\n");

    const llama_moe_status all[] = {
        LLAMA_MOE_STATUS_OK,
        LLAMA_MOE_STATUS_INVALID_CONFIG,
        LLAMA_MOE_STATUS_INVALID_EXPERT,
        LLAMA_MOE_STATUS_SLOTS_EXHAUSTED,
        LLAMA_MOE_STATUS_IO_ERROR,
        LLAMA_MOE_STATUS_SHORT_READ,
        LLAMA_MOE_STATUS_BACKEND_ERROR,
        LLAMA_MOE_STATUS_CANCELLED,
    };

    for (const auto s : all) {
        const char * str = llama_moe_status_str(s);
        CHECK(str != nullptr);
        CHECK(str[0] != '\0');
    }
}

//
// reader tests
//

static const size_t TEST_FILE_SIZE = 64 * 1024;

static uint8_t pattern_at(size_t i) {
    return (uint8_t) ((i * 31u + 7u) & 0xFF);
}

// a temporary file filled with a deterministic pattern; null if the platform refuses to make one
static FILE * make_pattern_file() {
    FILE * f = tmpfile();
    if (f == nullptr) {
        return nullptr;
    }

    std::vector<uint8_t> buf(TEST_FILE_SIZE);
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = pattern_at(i);
    }

    if (fwrite(buf.data(), 1, buf.size(), f) != buf.size() || fflush(f) != 0) {
        fclose(f);
        return nullptr;
    }

    return f;
}

static void test_reader_reads(int32_t n_threads) {
    printf("test_reader_reads (n_threads=%d)\n", n_threads);

    FILE * f = make_pattern_file();
    if (f == nullptr) {
        printf("  SKIP: could not create a temporary file\n");
        return;
    }

    {
        llama_moe_reader reader;
        CHECK(reader.start_workers(n_threads) == LLAMA_MOE_STATUS_OK);

        size_t idx = 999;
        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE, &idx) == LLAMA_MOE_STATUS_OK);
        CHECK(idx == 0);
        CHECK(reader.n_files() == 1);
        CHECK(reader.file_size(0) == TEST_FILE_SIZE);
        CHECK(reader.file_size(7) == 0);

        // a batch of disjoint chunks, mimicking several experts read at once
        const size_t chunk = 1024;
        const size_t n     = 16;

        std::vector<uint8_t>            dst(chunk * n, 0);
        std::vector<llama_moe_read_req> reqs;

        for (size_t i = 0; i < n; i++) {
            // read them out of order so a seek-based implementation would trip
            const size_t src = ((n - 1) - i) * chunk;
            reqs.push_back({ 0, (uint64_t) src, dst.data() + i * chunk, chunk });
        }

        CHECK(reader.read_many(reqs.data(), reqs.size()) == LLAMA_MOE_STATUS_OK);

        for (size_t i = 0; i < n; i++) {
            const size_t src = ((n - 1) - i) * chunk;
            for (size_t j = 0; j < chunk; j++) {
                if (dst[i * chunk + j] != pattern_at(src + j)) {
                    printf("  FAIL: chunk %zu byte %zu mismatched\n", i, j);
                    n_fail++;
                    j = chunk; // report once per chunk
                    i = n;
                }
            }
        }

        const auto st = reader.stats();
        CHECK(st.n_read  == n);
        CHECK(st.n_bytes == chunk * n);

        reader.reset_stats();
        CHECK(reader.stats().n_read  == 0);
        CHECK(reader.stats().n_bytes == 0);
    }

    fclose(f);
}

static void test_reader_bounds() {
    printf("test_reader_bounds\n");

    FILE * f = make_pattern_file();
    if (f == nullptr) {
        printf("  SKIP: could not create a temporary file\n");
        return;
    }

    {
        llama_moe_reader reader;
        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE, nullptr) == LLAMA_MOE_STATUS_OK);

        std::vector<uint8_t> dst(256, 0);

        // reading right up to the end is fine
        CHECK(reader.read({ 0, TEST_FILE_SIZE - 256, dst.data(), 256 }) == LLAMA_MOE_STATUS_OK);

        // ...but anything that would run past it is refused before the read is issued
        CHECK(reader.read({ 0, TEST_FILE_SIZE - 255, dst.data(), 256 }) == LLAMA_MOE_STATUS_IO_ERROR);
        CHECK(reader.read({ 0, TEST_FILE_SIZE,       dst.data(), 256 }) == LLAMA_MOE_STATUS_IO_ERROR);
        CHECK(reader.read({ 0, TEST_FILE_SIZE + 1,   dst.data(), 256 }) == LLAMA_MOE_STATUS_IO_ERROR);

        // an offset that overflows when the size is added must not wrap into a valid range
        CHECK(reader.read({ 0, UINT64_MAX - 16, dst.data(), 256 }) == LLAMA_MOE_STATUS_IO_ERROR);

        // unknown file, null destination
        CHECK(reader.read({ 3, 0, dst.data(), 256 }) == LLAMA_MOE_STATUS_INVALID_CONFIG);
        CHECK(reader.read({ 0, 0, nullptr,    256 }) == LLAMA_MOE_STATUS_INVALID_CONFIG);

        // a zero-length read is trivially satisfied
        CHECK(reader.read({ 0, 0, dst.data(), 0 }) == LLAMA_MOE_STATUS_OK);
    }

    fclose(f);
}

static void test_reader_short_read() {
    printf("test_reader_short_read\n");

    FILE * f = make_pattern_file();
    if (f == nullptr) {
        printf("  SKIP: could not create a temporary file\n");
        return;
    }

    {
        // claim the file is bigger than it is, so the bounds check passes but the read comes up short -
        // this is what a truncated or concurrently replaced model file looks like
        llama_moe_reader reader;
        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE * 2, nullptr) == LLAMA_MOE_STATUS_OK);

        std::vector<uint8_t> dst(4096, 0);

        CHECK(reader.read({ 0, TEST_FILE_SIZE - 1024, dst.data(), 4096 }) == LLAMA_MOE_STATUS_SHORT_READ);

        // a short read must not be counted as delivered bytes
        CHECK(reader.stats().n_read  == 0);
        CHECK(reader.stats().n_bytes == 0);
    }

    fclose(f);
}

static void test_reader_batch_error_propagates() {
    printf("test_reader_batch_error_propagates\n");

    FILE * f = make_pattern_file();
    if (f == nullptr) {
        printf("  SKIP: could not create a temporary file\n");
        return;
    }

    {
        llama_moe_reader reader;
        CHECK(reader.start_workers(4) == LLAMA_MOE_STATUS_OK);
        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE, nullptr) == LLAMA_MOE_STATUS_OK);

        std::vector<uint8_t>            dst(8 * 1024, 0);
        std::vector<llama_moe_read_req> reqs;

        for (size_t i = 0; i < 8; i++) {
            reqs.push_back({ 0, (uint64_t) (i * 1024), dst.data() + i * 1024, 1024 });
        }

        // one bad request anywhere in the batch fails the whole batch
        reqs[5].offset = TEST_FILE_SIZE + 4096;

        CHECK(reader.read_many(reqs.data(), reqs.size()) == LLAMA_MOE_STATUS_IO_ERROR);

        // and the workers are still usable afterwards
        reqs[5].offset = 5 * 1024;
        CHECK(reader.read_many(reqs.data(), reqs.size()) == LLAMA_MOE_STATUS_OK);
    }

    fclose(f);
}

static void test_reader_lifecycle() {
    printf("test_reader_lifecycle\n");

    // no path, no file
    {
        llama_moe_reader reader;
        CHECK(reader.add_path(nullptr, nullptr) == LLAMA_MOE_STATUS_INVALID_CONFIG);
        CHECK(reader.add_path("./definitely-not-a-real-model-file.gguf", nullptr) == LLAMA_MOE_STATUS_IO_ERROR);
        CHECK(reader.n_files() == 0);

        CHECK(reader.add_fd(-1, 1024, nullptr) == LLAMA_MOE_STATUS_INVALID_CONFIG);
        CHECK(reader.add_fd(0,     0, nullptr) == LLAMA_MOE_STATUS_INVALID_CONFIG);
    }

    FILE * f = make_pattern_file();
    if (f == nullptr) {
        printf("  SKIP: could not create a temporary file\n");
        return;
    }

    {
        llama_moe_reader reader;
        CHECK(reader.start_workers(3) == LLAMA_MOE_STATUS_OK);
        CHECK(reader.n_threads() == 3);

        // workers may only be started once
        CHECK(reader.start_workers(2) == LLAMA_MOE_STATUS_INVALID_CONFIG);

        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE, nullptr) == LLAMA_MOE_STATUS_OK);

        std::vector<uint8_t>            dst(1024, 0);
        std::vector<llama_moe_read_req> reqs = { { 0, 0, dst.data(), 1024 } };

        CHECK(reader.read_many(reqs.data(), reqs.size()) == LLAMA_MOE_STATUS_OK);
        CHECK(reader.read_many(nullptr, 0) == LLAMA_MOE_STATUS_OK);
        CHECK(reader.read_many(nullptr, 1) == LLAMA_MOE_STATUS_INVALID_CONFIG);

        // after shutdown, further work is refused rather than silently skipped
        reader.shutdown();
        CHECK(reader.read_many(reqs.data(), reqs.size()) == LLAMA_MOE_STATUS_CANCELLED);

        // shutdown is idempotent
        reader.shutdown();
    }

    // the reader duplicated the descriptor, so closing ours does not disturb it
    {
        llama_moe_reader reader;
        CHECK(reader.add_fd(fileno(f), TEST_FILE_SIZE, nullptr) == LLAMA_MOE_STATUS_OK);

        fclose(f);
        f = nullptr;

        std::vector<uint8_t> dst(1024, 0);
        CHECK(reader.read({ 0, 0, dst.data(), 1024 }) == LLAMA_MOE_STATUS_OK);

        for (size_t j = 0; j < 1024; j++) {
            if (dst[j] != pattern_at(j)) {
                printf("  FAIL: byte %zu mismatched after the original handle was closed\n", j);
                n_fail++;
                break;
            }
        }
    }

    if (f != nullptr) {
        fclose(f);
    }
}

int main() {
    test_min_slots();
    test_init_validation();
    test_miss_then_hit();
    test_repeated_id_fills_once();
    test_large_id_space_stays_sparse();
    test_lfu_eviction_order();
    test_recency_breaks_frequency_ties();
    test_in_ubatch_pinning();
    test_minimum_slot_boundary();
    test_invalid_expert_id();
    test_empty_and_bad_arguments();
    test_invalidate();
    test_stats_accounting();
    test_status_strings();

    test_reader_reads(0);
    test_reader_reads(1);
    test_reader_reads(4);
    test_reader_bounds();
    test_reader_short_read();
    test_reader_batch_error_propagates();
    test_reader_lifecycle();

    if (n_fail > 0) {
        printf("\n%d check(s) failed\n", n_fail);
        return 1;
    }

    printf("\nall checks passed\n");
    return 0;
}
