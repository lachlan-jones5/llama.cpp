// Greedy speculative decoding through the MTP draft head must reproduce plain greedy decoding.
//
// Speculative decoding is defined to preserve the target distribution: whatever the drafter proposes, the
// target verifies every token, so the greedy sequence must come out the same as it would without a drafter.
// That makes greedy equality the one property of the whole draft loop that can be asserted without knowing
// anything about the head's quality. Run against the generated qwen4exp model, whose fixture carries a head.
//
// The comparison stops at the first token that differs only if the plain run's top-1/top-2 margin is within
// the logit drift between the two runs: a tie broken by batch-shape numerics, not a wrong head. A random
// fixture has near-flat logits, so that case is real; a wrong head shows up as a margin the drift cannot
// explain, or as the drafter never being accepted at all.
//
// Usage: test-speculative-mtp -m <model.gguf>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-ext.h"

static int n_fail = 0;

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                \
        }                                                            \
    } while (0)

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double e = 0.0, r = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        e += d*d;
        r += (double) a[i] * (double) a[i];
    }
    return r == 0.0 ? (e == 0.0 ? 0.0 : INFINITY) : e / r;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

// gap between the best and the second-best logit: how decisively the plain run chose its token
static double top_margin(const std::vector<float> & logits) {
    float best = -INFINITY, second = -INFINITY;
    for (float v : logits) {
        if (v > best) { second = best; best = v; }
        else if (v > second) { second = v; }
    }
    return (double) best - (double) second;
}

struct step {
    llama_token        id;
    std::vector<float> logits;  // the row that produced id
};

static std::vector<float> row(llama_context * ctx, int32_t i, uint32_t n_vocab) {
    const float * p = llama_get_logits_ith(ctx, i);
    return std::vector<float>(p, p + n_vocab);
}

static llama_context_params make_cparams(uint32_t n_ctx, uint32_t n_rs_seq) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = 32;
    cp.n_ubatch  = 32;
    cp.n_rs_seq  = n_rs_seq;
    cp.no_perf   = true;
    cp.n_seq_max = 1;
    return cp;
}

static common_params_sampling greedy() {
    common_params_sampling sp;
    sp.temp     = 0.0f;
    sp.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sp.seed     = 1;
    return sp;
}

// plain greedy: one token at a time, no drafter
static std::vector<step> run_plain(llama_model * model, const llama_tokens & prompt, int n_gen, uint32_t n_ctx) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_ptr ctx(llama_init_from_model(model, make_cparams(n_ctx, 0)));
    CHECK(ctx != nullptr);
    if (!ctx) return {};

    common_params_sampling sparams = greedy();
    common_sampler_ptr smpl(common_sampler_init(model, sparams));

    std::vector<step> out;

    llama_batch batch = llama_batch_init(32, 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], i, {0}, i + 1 == prompt.size());
    }
    CHECK(llama_decode(ctx.get(), batch) == 0);

    int n_past = prompt.size();
    for (int k = 0; k < n_gen; k++) {
        const int32_t idx = batch.n_tokens - 1;
        step s;
        s.logits = row(ctx.get(), idx, n_vocab);
        s.id     = common_sampler_sample(smpl.get(), ctx.get(), idx);
        common_sampler_accept(smpl.get(), s.id, true);
        out.push_back(std::move(s));

        common_batch_clear(batch);
        common_batch_add(batch, out.back().id, n_past++, {0}, true);
        CHECK(llama_decode(ctx.get(), batch) == 0);
    }
    llama_batch_free(batch);
    return out;
}

// greedy through the MTP drafter: the loop examples/speculative-simple runs, checkpoints included
static std::vector<step> run_spec(llama_model * model, const llama_tokens & prompt, int n_gen, uint32_t n_ctx,
                                  int * n_drafted_out, int * n_accepted_out) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_draft_max = 4;

    llama_context_ptr ctx_tgt(llama_init_from_model(model, make_cparams(n_ctx, 0)));
    CHECK(ctx_tgt != nullptr);
    if (!ctx_tgt) return {};

    common_params params;
    params.n_ctx    = n_ctx;
    params.n_batch  = 32;
    params.n_ubatch = 32;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = n_draft_max;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;
    params.speculative.draft.backend_sampling = false;
    params.speculative.draft.ctx_tgt = ctx_tgt.get();

    // creates the MTP draft context against the target model
    auto spec_init = common_speculative_init_from_params(params, model, ctx_tgt.get());
    llama_context * ctx_dft = spec_init ? spec_init->context() : nullptr;
    CHECK(ctx_dft != nullptr);
    if (!ctx_dft) return {};
    params.speculative.draft.ctx_dft = ctx_dft;

    // The head-only memory holds positions and no recurrent state, so the draft context must support
    // partial sequence removal outright rather than needing a checkpoint every round. The target's trunk
    // is recurrent and qwen4exp does not (yet) support rollback snapshots, so it takes the checkpoint path.
    const auto rm_dft = common_context_can_seq_rm(ctx_dft);
    const auto rm_tgt = common_context_can_seq_rm(ctx_tgt.get());
    auto rm_name = [](common_context_seq_rm_type t) {
        return t == COMMON_CONTEXT_SEQ_RM_TYPE_PART ? "partial" :
               t == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ? "full only (checkpoints)" :
               t == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ? "rollback snapshots" : "none";
    };
    printf("  seq_rm support: target %s, draft %s\n", rm_name(rm_tgt), rm_name(rm_dft));
    CHECK(rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_PART);

    const bool use_ckpt_tgt = rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    // the probes above decoded into both contexts; start clean
    llama_memory_clear(llama_get_memory(ctx_tgt.get()), true);
    llama_memory_clear(llama_get_memory(ctx_dft),       true);

    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    CHECK(spec != nullptr);
    if (!spec) return {};

    common_params_sampling sparams = greedy();
    common_sampler_ptr smpl(common_sampler_init(model, sparams));

    const llama_seq_id seq_id = 0;
    std::vector<step> out;

    // the prompt minus its last token goes through the target and the drafter together
    {
        llama_batch b = llama_batch_init(32, 0, 1);
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            common_batch_add(b, prompt[i], i, { seq_id }, false);
        }
        CHECK(llama_decode(ctx_tgt.get(), b) == 0);
        CHECK(common_speculative_process(spec.get(), b));
        llama_batch_free(b);
    }

    llama_token  id_last = prompt.back();
    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    int n_past = prompt_tgt.size();

    common_speculative_begin(spec.get(), seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(32, 0, 1);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    int n_drafted = 0, n_accepted = 0;

    // Only a draft the head produced counts. On the checkpoint path a partial acceptance re-verifies the
    // accepted prefix as the next "draft", and the target agreeing with its own sampled tokens says
    // nothing about the head - counting those would report 100 percent for a drafter that proposes junk.
    bool draft_from_head = false;

    while ((int) out.size() < n_gen) {
        if (draft.empty()) {
            draft_from_head = true;
            ckpt.update_pos(prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt.get()), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt.get()), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            int n_draft_budget = std::min(n_draft_max, n_gen - (int) out.size() - 1);
            n_draft_budget = std::max(n_draft_budget, 0);

            common_speculative_get_draft_params(spec.get(), seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ n_draft_budget,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(spec.get());

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // drafting ran the head forward past n_past; rewind it before the verify batch is replayed
            // through it. This is a partial removal on a state-less recurrent cache, the case the memory
            // fix exists for.
            if (use_ckpt_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1));
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); i++) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }
        CHECK(llama_decode(ctx_tgt.get(), batch_tgt) == 0);
        CHECK(common_speculative_process(spec.get(), batch_tgt));

        // the rows are read before sampling accepts and advances anything
        std::vector<std::vector<float>> rows;
        for (int32_t i = 0; i < batch_tgt.n_tokens; i++) {
            rows.push_back(row(ctx_tgt.get(), i, n_vocab));
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt.get(), draft);
        CHECK(!ids.empty());

        if (draft_from_head) {
            n_drafted  += n_draft;
            n_accepted += ids.size() - 1;
            draft_from_head = false;
        }

        // partial acceptance without partial removal: restore, and re-verify the accepted prefix as the
        // next draft, exactly as the example does
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            draft = std::move(ids);

            ckpt.load_tgt(ctx_tgt.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), seq_id, ckpt.pos_max + 1, -1));

            if (use_ckpt_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1));

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec.get(), seq_id, ids.size() - 1);

        n_past     += ids.size() - 1;

        for (size_t i = 0; i < ids.size() && (int) out.size() < n_gen; i++) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            out.push_back({ ids[i], std::move(rows[i]) });
        }
        draft.clear();

        CHECK(llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), seq_id, n_past, -1));
        CHECK(llama_memory_seq_rm(llama_get_memory(ctx_dft),       seq_id, n_past, -1));

        // the drafter's own bookkeeping must track the accepted prefix exactly
        CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) == n_past - 1);
    }
    llama_batch_free(batch_tgt);

    *n_drafted_out  = n_drafted;
    *n_accepted_out = n_accepted;
    return out;
}

int main(int argc, char ** argv) {
    std::string path;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            path = argv[++i];
        } else {
            printf("usage: %s -m <model.gguf>\n", argv[0]);
            return 1;
        }
    }
    if (path.empty()) {
        printf("usage: %s -m <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    CHECK(model != nullptr);
    if (!model) {
        printf("cannot load %s\n", path.c_str());
        return 1;
    }
    CHECK(llama_model_n_layer_nextn(model) > 0);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx   = 64;
    const int      n_gen   = 24;

    // the generated models carry no vocab, so the prompt is plain in-range ids
    llama_tokens prompt;
    for (int i = 0; i < 8; i++) {
        prompt.push_back((llama_token) ((i * 7 + 1) % n_vocab));
    }

    printf("model: %s\n", path.c_str());

    printf("plain greedy\n");
    const auto plain = run_plain(model, prompt, n_gen, n_ctx);
    CHECK((int) plain.size() == n_gen);

    printf("speculative greedy through the MTP head\n");
    int n_drafted = 0, n_accepted = 0;
    const auto spec = run_spec(model, prompt, n_gen, n_ctx, &n_drafted, &n_accepted);
    CHECK((int) spec.size() == n_gen);
    printf("  drafted %d, accepted %d\n", n_drafted, n_accepted);

    // a drafter that never proposes anything would make the comparison vacuous
    CHECK(n_drafted > 0);

    printf("plain:");
    for (const auto & st : plain) printf(" %d", st.id);
    printf("\nspec: ");
    for (const auto & st : spec)  printf(" %d", st.id);
    printf("\n");

    printf("plain:");
    for (const auto & st : plain) printf(" %d", st.id);
    printf("\nspec: ");
    for (const auto & st : spec)  printf(" %d", st.id);
    printf("\n");

    printf("comparison\n");
    int n_same = 0;
    for (int k = 0; k < n_gen && k < (int) plain.size() && k < (int) spec.size(); k++) {
        const double err   = nmse(plain[k].logits, spec[k].logits);
        const double drift = max_abs_diff(plain[k].logits, spec[k].logits);

        if (plain[k].id == spec[k].id) {
            CHECK(err < 1e-4);
            n_same++;
            continue;
        }

        // the sequences diverge here; it is only acceptable if the plain choice was a near-tie the drift
        // could have flipped
        const double margin = top_margin(plain[k].logits);
        printf("  step %d: tokens differ (%d vs %d); plain top margin %.3e, logit drift %.3e\n",
                k, plain[k].id, spec[k].id, margin, drift);
        CHECK(margin <= 2.0 * drift + 1e-6);
        break;
    }
    printf("  %d of %d tokens identical before any divergence\n", n_same, n_gen);
    CHECK(n_same >= n_gen / 2);  // a wrong head diverges almost immediately

    llama_model_free(model);
    llama_backend_free();

    if (n_fail > 0) {
        printf("\n%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
