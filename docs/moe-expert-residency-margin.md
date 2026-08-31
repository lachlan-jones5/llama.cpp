# Plan: a memory margin that survives paged reads

Status: **not implemented, and deliberately not yet designed in detail.** We have one observation, and one
observation is not a margin policy. This records what is known, what the competing explanations are, and
which measurement distinguishes them.

## Why

`--moe-n-slots auto` picks the largest slot count that fits, using per-device free memory less a margin. On
the development host that works. On the 24 GiB Mac it is not yet safe, because a configuration that device
accounting said would fit **took 2,657 MiB of swap and 182,368 swapouts**:

| Configuration | Metal in use | Free after | Result |
| --- | ---: | ---: | --- |
| 32 slots, 40,960 ctx | 10,681 MiB | 7,503 MiB (31 %) | passed, no new swapouts |
| 64 slots | ~17,400 MiB | ~800 MiB | **swapped 2.7 GiB** |

The projection was not obviously wrong; it simply measured the wrong thing. So the acceptance criterion has
to be *no new swapouts*, and the fit cannot observe swap. It must therefore be conservative by construction,
and the question is by how much and as a function of what.

`--fit-target` already lets a user raise the margin by hand, so nobody is stuck while this is open.

## Two explanations, and they imply different policies

This is the part to settle before writing code.

**A. Page-cache competition.** With `--load-mode none` the model file is streamed through the page cache,
which on a unified-memory host contends with everything else. A prefill moved 1.16 TiB of expert bytes, so
the cache is under continuous pressure to hold as much of the file as it can. If this is the mechanism, the
margin should scale with **model file size or read volume**, and a small machine running a large model needs
a proportionally larger reserve.

**B. Device-reported free memory over-states what is usable.** At 64 slots the pools alone are roughly
13.3 GiB; adding KV at 40,960 tokens (3.8 GiB, at the measured 96 MiB per 1,024 tokens) plus compute and host
allocations gives ~17.5 GiB of *anonymous* memory on a 24 GiB machine. The OS needs several GiB of that for
itself. If this is the mechanism, page cache is a red herring, the reserve should scale with **total device
memory**, and the 31 % that passed is roughly the right shape.

These are not exotic alternatives — they predict different things and the data we asked for separates them:

- If the safe threshold tracks **total memory** (~31 % regardless of model), B is dominant → reserve a
  fraction of device total.
- If it tracks **file size or bytes read**, A is dominant → reserve a function of the model, and a 45 GiB
  model on the same host needs more headroom than a 25 GiB one.
- If both matter, the reserve is `max(fraction_of_total, f(file_size))`, and we will have the two points
  needed to fit it.

## The measurement that decides it

Requested from the Mac on tip `2db74dae4`: run `--moe-n-slots auto` on the 30B at 40,960 context, and report
the chosen count, the breakdown, whether swap moved, and **how much free memory remained**. Then the same on
Qwen3-Coder-Next 80B-A3B, which is 45 GiB of file against the 30B's 25 GiB on the same host — that pair
varies file size while holding the machine constant, which is exactly the discriminator between A and B.

Until those land, any constant written into the code is invented precision.

## Design sketch, to be confirmed by the above

- The reserve applies **only when paging is active**. A non-paged model does not stream a file through the
  cache and should not pay for it.
- Express it as an additional term in the existing per-device `margins`, so `--fit-target` continues to
  override and the fit's arithmetic is untouched.
- Log the chosen slot count *and* the reserve that produced it. When a user's machine swaps anyway, the
  first question is which term was too small, and that should be answerable from the log rather than by
  rebuilding.
- Prefer under-using memory to swapping. Swap on a paged model is pathological: the OS pages out the pools
  we are using to cache experts we are paging in from disk. Losing a few slots costs a fraction of prefill;
  swapping costs everything.

## Verification

- The Mac's own failure case: `auto` on the 30B at 40,960 context on the 24 GiB host must produce **zero new
  swapouts**. Swap counters are the evidence, not the projection agreeing with itself.
- The same on the 80B-A3B, where the file is 45 GiB — if the reserve is a function of total memory only, this
  is where that assumption breaks.
- A large-memory host must not be penalised into a needlessly small pool; check the chosen count against what
  a hand search finds on the development box.
- An explicit `--fit-target` still overrides in both directions.

## Risks

| Risk | Handling |
| --- | --- |
| Encoding a constant fitted to one host | Do not; wait for the second model on the same machine |
| Over-reserving and wasting a large machine's memory | Verify the chosen count against a hand search where memory is plentiful |
| Under-reserving and swapping | Bias toward under-use; the cost is asymmetric and swap is pathological here |
| Mechanism A and B confounded in one number | The 25 GiB vs 45 GiB pair on one host separates them |
