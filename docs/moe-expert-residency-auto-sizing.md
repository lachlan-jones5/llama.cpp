# Choosing the slot count automatically — design record

Status: **implemented** as `--moe-n-slots auto`. User-facing behaviour is documented in
`moe-expert-residency.md`; this file keeps the reasoning behind it, including the parts that were rejected.

Two things changed between this design and what shipped, both worth knowing:

- **Per-slot cost is measured, not derived.** The plan said to compute it from `llama_moe_tensor_info::stride`,
  the on-disk expert size. That is *not* the pool's slot stride — the pool tensor's `nb[2]` can differ
  through alignment — so it would have under-estimated, and under-estimating is exactly what causes swap.
  Two probes and an interpolation give the real figure per device instead, and need no new accessor.
- **The graph floor is needed for more than the clamp.** The first probe has to sit above it. Probing at
  `n_expert_used` with a large microbatch gives a group of one token, hundreds of groups per layer, and a
  context that fails to build for a reason unrelated to memory. `llama_moe_min_slots_for_graph()` exists
  because of that failure, not in anticipation of it.

## Why

`--moe-n-slots N` is the one number a user has to supply, and getting it right requires knowing three
interacting constraints that are nowhere visible from the command line:

- **A floor.** `N >= n_expert_used`, or a single token cannot be served.
- **A ceiling that is not the obvious one.** `N <= n_expert`, but the *useful* range ends earlier: once
  `n_expert_used * (N / n_expert_used)` reaches `n_expert`, everything is resident and paging buys nothing.
- **A second floor, from the graph budget, that moves in the opposite direction to intuition.** The expert
  path runs in groups of `N / n_expert_used` tokens, so a *smaller* N means *more* groups, and enough groups
  exceed the node budget and are refused. `--moe-n-slots 8 --ubatch-size 512` is rejected for this reason.

So the valid range is bounded below by two different things and above by memory, and a user who picks badly
gets either a refusal at startup or a configuration that quietly buys nothing. Meanwhile the information
needed to choose well is now all available to the fit pass — pools have been charged to the memory breakdown
since `f4d4ac65a`, which is what makes this worth doing now.

The goal: **give paging a memory budget and let it size itself.**

## What the fit already does, and the convention to respect

`common_fit_params` (`common/fit.cpp:878`) adjusts model and context parameters to fit free device memory.
Its documented rule is that it modifies **only parameters the user left at their default**, with context
size as a special case. That convention is exactly right here and must be kept: an explicit
`--moe-n-slots 32` is a user decision and stays untouched.

That creates a problem, though. `n_slots == 0` is both the default *and* the encoding for "paging off". The
fit cannot distinguish "user did not ask" from "user wants no paging", and silently enabling paging would
violate the standing rule against changing configuration to recover.

**Resolution: a third state.** `--moe-n-slots auto` sets a sentinel (`-1`) meaning *paging is on, you pick
the number*. Default stays `0`, meaning off. Paging remains strictly opt-in; only the size becomes
automatic.

## The constraints, as arithmetic

Given `n_expert`, `n_expert_used`, `n_layer_paged`, `n_ubatch` and a memory budget `B`:

```
floor_token  = n_expert_used                                  # one token's worth
floor_graph  = smallest N such that the node budget fits:
                 chunk    = N / n_expert_used
                 n_groups = ceil(n_ubatch / chunk)
                 nodes    = n_groups * (64 + 4*n_expert_used) * n_layer_all + n_ubatch*8
                 nodes   <= LLAMA_MOE_MAX_GRAPH_NODES
ceiling_mem  = largest N such that pool_bytes(N) <= B
ceiling_use  = n_expert                                       # beyond this is full residency
```

and the answer is `min(ceiling_mem, ceiling_use)`, rejected if that is below
`max(floor_token, floor_graph)`.

**Pools scale exactly linearly in N**, which makes `ceiling_mem` cheap to compute: one probe at a known slot
count gives the per-slot cost, and everything else is division. `pool_bytes(N) = N * sum(stride) *
n_layer_paged`, where `stride` is the per-expert byte size already recorded in `llama_moe_tensor_info`.

### Objective: take the largest N that fits

More slots means a higher hit rate and fewer bytes read. The Metal measurements support this directly —
prefill 10.49 / 11.59 / 13.81 t/s at 8 / 16 / 32 slots, with hit rate 37.7 / 63.3 / 79.0 %.

One honest caveat to carry into the implementation: on the Linux/NVMe host, going 8 -> 32 slots more than
doubled the hit rate and read 40 % fewer bytes yet left *cold* throughput unchanged. Slot count buys memory
predictability and helps most where reads are a large share of wall time. "Largest that fits" is still the
right default, but it is not a guarantee of speed on every host.

## Mechanism

The fit already builds a `no_alloc` probe context to measure memory. Extend that path:

1. Run the existing probe with a provisional slot count — `n_expert_used`, the smallest legal value, so the
   probe itself cannot fail the graph budget for a reason unrelated to memory.
2. Read pool bytes out of the breakdown (`llama_moe_residency::memory_breakdown()`, already keyed by buffer
   type, so per-device budgets work).
3. Divide to get bytes-per-slot, solve for `ceiling_mem` against the free memory the fit already computes.
4. Clamp to the range above; if it is empty, fail with a message naming which bound bit and what to change —
   the same shape as the existing graph-budget refusal, which names an exact microbatch.

Files this touches: `common/fit.cpp` (the search), `common/arg.cpp` (parsing `auto`), `include/llama.h`
(the sentinel's meaning), `src/llama-context.cpp` (accepting a resolved count), and the existing validation
in `src/llama.cpp:319`.

### Interaction with layer offload — a deliberate simplification

The fit's existing job is choosing `n_gpu_layers`, and paging changes the memory each layer costs, so in
principle slot count and layer count are a joint optimisation. **Do not attempt that.** Size the slot count
once against the budget, then let the existing layer search run with pool memory included, as it already
does. A joint search is a lot of machinery for a second-order gain, and the simple version is explainable to
a user, which the joint one is not.

## Commit plan

1. `feat(moe): accept --moe-n-slots auto` — parsing, the sentinel, validation. Behaves as today otherwise.
2. `feat(moe): report bytes-per-slot from the residency manager` — the coefficient the fit needs.
3. `feat(fit): choose a slot count that fits the memory budget` — the search and the clamping.
4. `fix(fit): explain an empty slot range` — the refusal path, naming the binding constraint.
5. `docs(moe): document automatic slot sizing`.

## Verification

- `--moe-n-slots auto` on a model that fits comfortably picks a count at or near `n_expert` and is
  bit-exact against an explicit run at the same count.
- On a memory-constrained device it picks a smaller count, and the resulting peak footprint is within the
  budget — checked against measured RSS, not just the projection.
- An explicit `--moe-n-slots N` is **never** modified, including when it does not fit; the existing failure
  path applies.
- `--moe-n-slots auto --ubatch-size 512` on a model whose graph floor exceeds its memory ceiling refuses
  with a message naming both bounds rather than picking something that then fails at startup.
- Default (`0`) still means paging off, with no probe cost and no behaviour change.
- Metal, CUDA and CPU all agree on the chosen count for the same budget, since the arithmetic is
  backend-independent — only the budget differs.

## Risks

| Risk | Handling |
| --- | --- |
| Silently enabling paging for users who did not ask | `auto` is opt-in; default `0` stays off |
| Overriding an explicit user choice | Follow the existing fit convention: only touch defaults |
| Projection disagrees with reality and the run swaps | Verify against measured RSS; the Mac's 12 GB swap storm is what this must never cause |
| Largest-that-fits is not fastest on slow storage | Documented; the CPU cold data shows slot count is not a reliable speed lever |
| Graph floor and memory ceiling cross | Explicit refusal naming both, in the style of the existing budget message |
