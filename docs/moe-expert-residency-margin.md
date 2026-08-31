# Plan: how aggressively should `auto` spend memory?

Status: **not implemented.** The mechanism to control this already exists (`--fit-target`); what is missing
is a sensible default when paging is enabled automatically, and that default cannot be chosen from the data
we have.

## Why this is not a safety gate

The first framing of this document was wrong. It treated any swap as failure and asked "what margin
guarantees no swapouts?". The actual criterion is whether the machine is **usable for the job it is doing**:

> A one-time displacement of inactive pages is acceptable. Sustained swap churn that stalls the agent is not.

That distinction matters because a large footprint is *the point* on a dedicated machine. Two profiles fall
out, and both are legitimate:

| Profile | Slots (30B, 24 GiB Mac) | Character |
| --- | ---: | --- |
| Conservative | 32 | Proven no new swapouts at 40,960 context, alongside other applications |
| Aggressive | 68 (what `auto` picks) | Loads and serves at 40,960 context; 64 slots measured **107.93 t/s** prefill |

`auto` choosing 68 is not a bug. It is the right answer for a dedicated host and the wrong one for a shared
desktop, and nothing in the memory numbers tells the fit which it is on.

## Why swapping *while paging* is worse than swapping normally

Worth stating because it is the reason sustained churn is disqualifying rather than merely slow. The pools
exist to cache expert weights read from disk. If the OS pages those pools out, it is writing to disk the very
data we are reading from disk to avoid reading it from disk. Under sustained pressure that is a doom loop,
and throughput collapses rather than degrading gracefully. One-time displacement of *other* processes' idle
pages has none of that character.

## What the measurements actually show

**Device-reported free memory over-states what is usable.** `auto` sized to a budget of 17,061 MiB and landed
at 17,060 MiB — the arithmetic was exact. But Metal reported roughly **18,085 MiB free on a 24,576 MiB
machine that already had 2,653 MiB in swap**. The number the fit is handed does not reserve for the host's
own needs, and on unified memory there is no second pool to fall back on.

**Read volume matters too, separately from the static allocation.** A 64-slot `pp8192` run swapped ~2.85 GiB
with a *smaller* context than the 68-slot run. Static allocation alone does not explain that; streaming 1.16
TiB of expert bytes through the page cache does.

So both mechanisms this document originally posed as alternatives are real, and a single constant fitted to
one host would encode neither.

## Design direction

- **Do not invent a constant.** The control already exists: `--fit-target` raises the reserve and is
  respected by the sizing arithmetic unchanged.
- **Apply any default reserve only when paging is active.** A non-paged model does not stream a file through
  the page cache and should not pay for it.
- **Default conservatively, let dedicated hosts opt into more.** A default that swaps a mixed desktop is a
  worse failure than one that leaves performance unclaimed, because the second is visible and tunable while
  the first looks like the machine is broken.
- **Log the reserve next to the chosen count.** When a host swaps anyway, the first question is which term
  was too small; that should be answerable from the log.
- Consider naming the profiles rather than exposing a byte count, since "dedicated machine" and "shared
  desktop" are the decision a user can actually make. Do not build this until the qualification below says
  what the two settings should be.

## The qualification that decides it

Not another synthetic no-swap gate. A real Claude Code session on a dedicated host at `auto`, with unrelated
applications closed, measuring:

1. startup and repository-context time to first token;
2. sustained generation speed and tool-call correctness;
3. **whether swapout activity stops after warmup or continues during work** — the distinction this whole
   document turns on;
4. system responsiveness and memory-pressure state during a multi-step agent task;
5. behaviour at the real 40,960-token context ceiling.

If swapouts stop after warmup, `auto`'s current aggressiveness is right for a dedicated host and the work is
to expose the choice. If they continue, the reserve needs to grow and (3) tells us by how much.

## Verification

- The chosen count and the reserve that produced it appear in the log at default verbosity.
- A large-memory host is not penalised into a needlessly small pool — check against a hand search where
  memory is plentiful.
- `--fit-target` overrides in both directions, and an explicit `--moe-n-slots N` is still never touched.
- Whatever default is chosen is re-checked on both the 25 GiB and 45 GiB models on the same host, since read
  volume is now known to matter independently of the static footprint.

## Risks

| Risk | Handling |
| --- | --- |
| Encoding a constant fitted to one host and one model | Wait for the qualification run; two models on one host at minimum |
| A default that quietly swaps a shared desktop | Default conservatively; unclaimed performance is visible and tunable, a thrashing machine is not |
| Treating any swap as failure and under-using a dedicated machine | The criterion is sustained churn, not the swap counter |
| Profiles becoming another knob nobody understands | Only add them if the qualification shows two genuinely different good answers |
