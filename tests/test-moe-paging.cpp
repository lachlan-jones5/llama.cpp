// Paged MoE experts must produce the same result as fully resident ones.
//
// Loads the same GGUF twice - once normally, once with a bounded expert pool - decodes identical tokens
// through both, and compares the logits. Also covers the slot boundary and the configurations that must be
// rejected rather than silently accepted.
//
// Usage: test-moe-paging -m <model.gguf> [--slots N]

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                \
        }                                                            \
    } while (0)

// normalized mean squared error, matching the tolerance test-llama-archs uses between backends
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }

    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];

        mse_a_b += d * d;
        mse_a_0 += (double) a[i] * (double) a[i];
    }

    if (mse_a_0 == 0.0) {
        return mse_a_b == 0.0 ? 0.0 : INFINITY;
    }

    return mse_a_b / mse_a_0;
}

struct run_result {
    bool               ok = false;
    std::vector<float> logits;
};

// Load the model with the given slot count (0 = paging off) and decode tokens, returning the logits.
static int32_t g_n_gpu_layers = 0;

static run_result run(const std::string & path, int32_t n_slots, uint32_t n_ubatch, bool expect_ok) {
    run_result out;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = g_n_gpu_layers;  // both runs use the same placement, so the comparison is fair
    mparams.moe.n_slots  = n_slots;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        if (expect_ok) {
            printf("  failed to load the model with %d slots\n", n_slots);
        }
        return out;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 64;
    cparams.n_batch  = 32;
    cparams.n_ubatch = n_ubatch;
    cparams.no_perf  = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        if (expect_ok) {
            printf("  failed to create a context with %d slots\n", n_slots);
        }
        llama_model_free(model);
        return out;
    }

    // the generated test models carry no vocab, so drive them with plain in-range token ids
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> tokens;
    for (uint32_t i = 0; i < 12; i++) {
        tokens.push_back((llama_token) ((i * 7 + 1) % n_vocab));
    }

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t pos = 0; pos < tokens.size(); pos++) {
        common_batch_add(batch, tokens[pos], (llama_pos) pos, {0}, true);
    }

    const int rc = llama_decode(ctx, batch);

    if (rc == 0) {
        out.ok = true;
        out.logits.reserve(tokens.size() * n_vocab);

        for (size_t i = 0; i < tokens.size(); i++) {
            const float * row = llama_get_logits_ith(ctx, (int32_t) i);
            for (uint32_t j = 0; j < n_vocab; j++) {
                out.logits.push_back(row[j]);
            }
        }
    } else if (expect_ok) {
        printf("  llama_decode failed with %d at %d slots\n", rc, n_slots);
    }

    if (n_slots > 0 && out.ok) {
        const auto st = llama_moe_stats(ctx);

        printf("  %3d slots: lookups=%lld hits=%lld misses=%lld evictions=%lld reads=%lld bytes=%lld\n",
                n_slots,
                (long long) st.n_lookup, (long long) st.n_hit, (long long) st.n_miss,
                (long long) st.n_evict, (long long) st.n_read, (long long) st.n_bytes_read);

        // paging must actually have done something, or the comparison below proves nothing
        CHECK(st.n_lookup > 0);
        CHECK(st.n_miss > 0);
        CHECK(st.n_bytes_read > 0);
        CHECK(st.n_hit + st.n_miss == st.n_lookup);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return out;
}

int main(int argc, char ** argv) {
    std::string path;
    int32_t     slots_arg = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            path = argv[++i];
        } else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            slots_arg = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--n-gpu-layers") == 0) && i + 1 < argc) {
            g_n_gpu_layers = atoi(argv[++i]);
        } else {
            printf("usage: %s -m <model.gguf> [--slots N] [-ngl N]\n", argv[0]);
            return 1;
        }
    }

    if (path.empty()) {
        printf("usage: %s -m <model.gguf> [--slots N]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        // keep the load chatter out of the way, but never hide errors
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    printf("model: %s (n_gpu_layers = %d)\n", path.c_str(), g_n_gpu_layers);

    // reference: everything resident
    printf("reference (no paging)\n");
    const run_result ref = run(path, 0, 1, true);
    CHECK(ref.ok);

    if (!ref.ok) {
        printf("\ncannot continue without a reference run\n");
        llama_backend_free();
        return 1;
    }

    // The generated MoE test models use n_expert = 2 with 1 used per token, so a single-token ubatch needs
    // exactly 1 slot and 2 slots means everything stays resident. Both must match the reference.
    const std::vector<int32_t> slot_counts = slots_arg > 0 ? std::vector<int32_t>{ slots_arg }
                                                           : std::vector<int32_t>{ 1, 2 };

    printf("paged\n");
    for (const int32_t n_slots : slot_counts) {
        const run_result paged = run(path, n_slots, 1, true);
        CHECK(paged.ok);

        if (!paged.ok) {
            continue;
        }

        const double err = nmse(ref.logits, paged.logits);
        printf("  %3d slots: nmse vs resident = %.3e\n", n_slots, err);

        // the same weights are multiplied in the same order, so this should be exact rather than merely close
        CHECK(err < 1e-6);
    }

    // Multi-token ubatches are the case that catches id-layout mistakes. ggml_argsort_top_k returns a view
    // strided by n_expert, so reading the selected ids as a flat array happens to work for a single token
    // and silently reads the wrong ids for several. Compare a paged multi-token ubatch against a resident
    // one to keep that from coming back.
    printf("multi-token ubatch\n");
    {
        const uint32_t n_ubatch = 8;

        const run_result ref_multi = run(path, 0, n_ubatch, true);
        CHECK(ref_multi.ok);

        // Slot counts below n_expert_used * n_ubatch used to be refused outright. The expert path is now
        // split into groups small enough for the pool, so they must work - and must still be exact.
        //
        // One slot with an 8-token microbatch is the hardest case: the pool holds a single expert, so every
        // group evicts the previous group's. If a later group's admission could disturb a group whose matmul
        // has not run yet, this is where it would show up as wrong logits rather than a crash.
        for (const int32_t n_slots : { 2, 1 }) {
            const run_result paged_multi = run(path, n_slots, n_ubatch, true);
            CHECK(paged_multi.ok);

            if (ref_multi.ok && paged_multi.ok) {
                const double err = nmse(ref_multi.logits, paged_multi.logits);
                printf("  ubatch %u, %d slots: nmse vs resident = %.3e\n", n_ubatch, n_slots, err);
                CHECK(err < 1e-6);
            }
        }
    }

    // A ubatch of N tokens can route to n_expert_used*N distinct experts, all of which must be resident at
    // once, so too few slots for the ubatch has to be refused before inference rather than silently wrong.
    printf("rejected configurations\n");
    {
        // more slots than the layer has experts is a configuration error
        const run_result too_many = run(path, 1024, 1, false);
        CHECK(!too_many.ok);
        printf("  1024 slots: %s\n", too_many.ok ? "accepted (BAD)" : "rejected");
    }

    llama_backend_free();

    if (n_fail > 0) {
        printf("\n%d check(s) failed\n", n_fail);
        return 1;
    }

    printf("\nall checks passed\n");
    return 0;
}
