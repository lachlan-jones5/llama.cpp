# MoE expert residency (expert paging)

Mixture-of-Experts models carry a lot of weight that any single token does not touch. A layer may hold 256
experts and route each token to 8 of them. Expert paging keeps only a bounded number of experts of each layer
resident and reads the rest from the GGUF as the router selects them.

This trades throughput for memory. It is off by default and only worth turning on in one situation: **the
model does not fit, but its working set does.**

## Options

| Option | Meaning |
| --- | --- |
| `--moe-n-slots N` | Keep `N` experts of each paged layer resident. `0` (default) disables paging entirely. `auto` pages and lets the fit pass choose `N`. |
| `--moe-n-layers N` | Page only the first `N` MoE layers. `0` (default) pages every MoE layer. |
| `--moe-chunk-size N` | Tokens per expert-matmul group. `0` (default) derives it as `n_slots / n_expert_used`. |
| `--moe-read-threads N` | Threads used to read experts. `0` (default) picks a sensible number. |

Each has a matching `LLAMA_ARG_*` environment variable.

### Prefer `--moe-n-slots auto`

The slot count is the single most consequential setting, and it is genuinely hard to choose by hand. On
Metal, Qwen3-30B-A3B at `pp8192 -ub 512`, going from 32 to 64 slots is **+80.7 % prefill and −78.8 % bytes
read** — more than any other change measured on this feature. It matters twice over: a bigger pool caches
more, *and* a bigger pool means a bigger group, so the microbatch is cut into fewer pieces and the same
expert is reloaded far less often.

`auto` asks the fit pass to pick it:

- If the model fits **without** paging, it leaves paging off. Paging a model that already fits only adds
  the grouped-execution cost — measured at **3.4× slower** on a fully hot pool, where no I/O happens at all.
- Otherwise it sizes the pool *before* reducing the context, since paging experts is the cheaper way to free
  memory and context is usually what you want to keep.
- It probes twice and interpolates. Pools are exactly linear in the slot count, so two measurements give the
  per-slot cost per device including alignment.
- It clamps to the range below, and refuses with both bounds named if that range is empty.

An explicit `--moe-n-slots N` is never overridden, following the fit's convention of only adjusting
parameters left at their default.

**One thing `auto` cannot see: page-cache pressure.** With `--load-mode none` the model file is streamed
through the page cache, which on unified memory competes with everything else. A 24 GiB Mac running 64 slots
took 2.7 GiB of swap even though device accounting said it fit; the configuration that passed had 31 % still
free. If a chosen count swaps, raise `--fit-target` and re-run — and treat swap counters, not the
projection, as the evidence.

### Choosing a slot count by hand

A group of `N` tokens can route to `n_expert_used * N` distinct experts, and every one of them has to be
resident at the same time. So the slot count sets the group size:

```
group = n_slots / n_expert_used          (or --moe-chunk-size, if given)
```

The slot count is therefore just a memory budget: how many experts of each layer you are willing to keep
resident. It does **not** constrain `--ubatch-size` or `--parallel` — the expert path is split into groups
that fit, so a large microbatch is fine. Only two things are refused, both at load with the numbers named:
fewer slots than `n_expert_used` (a single token would not fit), and an explicit `--moe-chunk-size` larger
than the pool can hold.

This was not always true. Before token grouping the bound applied to the whole microbatch, so
`--moe-n-slots 8` required `--ubatch-size 1` and `--parallel 1`, and a 4-slot server needed 32 experts
resident. If you are reading older notes or configurations, that is why.

### Token groups: the microbatch is no longer what the slots bound

The expert path runs in **groups** of `--moe-chunk-size` tokens (default `n_slots / n_expert_used`).
Attention and the dense path still see the whole microbatch; only the expert matmuls are grouped. Since the
MoE FFN is per-token this is a regrouping of identical arithmetic, and it is verified bit-exact.

The consequence is that the bound below applies to the **group**, not the microbatch, so the slot count no
longer has to scale with `--ubatch-size` or `--parallel`. What replaces it is a limit on the number of
groups rather than on the microbatch: a slot count small enough to give a group of one token needs one group
per token, and enough of those exceed the graph node budget and are refused with the microbatch that would
fit (see *A small slot count with a large microbatch* below).

**The throughput gain from this is modest — measure before relying on it.** On CUDA at full GPU residency,
32 slots, `pp256`:

| Microbatch | `pp256` | vs the old ceiling | Graph nodes | Splits |
| ---: | ---: | ---: | ---: | ---: |
| 4 (the old ceiling) | 36.77 | — | 3,807 | 82 |
| 64 | 39.29 | +6.9 % | 20,207 | 1,282 |
| 512 | 40.10 | +9.1 % | 72,047 | 5,122 |

Only about an eighth of the available headroom (fully resident at microbatch 512 reaches 65.46). The reason
is visible in the last two columns: each group needs its own host-side resolve, which the scheduler turns
into a pair of graph splits with a device synchronisation, so the split count grows with the number of
groups and eats most of the batching win. That is also why the gain flattens almost immediately —
microbatch 64 already captures three quarters of what 512 does.

**On Metal the same change is worth far more.** Validated on an M5 Pro with Qwen3-30B-A3B-Q6_K, `pp32`,
microbatch 64 against microbatch 1:

| Slots | Warm prefill | Cold prefill |
| ---: | ---: | ---: |
| 8 | +24.5 % | +19.2 % |
| 16 | +35.0 % | +36.2 % |
| 32 | **+54.1 %** | **+48.5 %** |

Generation is unchanged on both backends, which is the expected shape — one token is one group and takes the
ungrouped path. Do not read the CUDA figure as the general case: the benefit depends on how much the
attention and dense path care about batch size on that backend, and Metal cares a great deal more. Both
measurements are small-`pp` tests; treat them as evidence that the effect is large and backend-dependent,
not as a number to quote for a long prompt.

**So prefer a modest microbatch.** 64 gets most of the benefit for 20k nodes; 512 costs 72k nodes, and at
roughly 21.5 KB per node — dominated by the scheduler's context buffer, not tensor overhead — that is about
1.5 GB of host memory against 250 MB.

**A small slot count with a large microbatch is refused, and the message says what to change.** The group is
`n_slots / n_expert_used`, so `--moe-n-slots 8` on an 8-expert-per-token model gives a group of one token and
needs one group per token: at `--ubatch-size 512` that is 512 groups in every layer, a two-million-node graph
budget, and — because the scheduler reserves its context buffer for one split per node at ~19.7 KiB each — an
allocation of about 40 GiB that simply fails. That used to surface as
`GGML_ASSERT(ctx->mem_buffer != NULL)` from inside `ggml_init`, which says nothing useful. It is now caught
before the scheduler is built:

```
expert paging: --moe-n-slots 8 gives a group of 1 token(s), so a microbatch of 512 needs 512 expert groups
and a graph budget of 2019328 nodes, past the 1048576-node limit. Either lower --ubatch-size to 265 or
below, or raise --moe-n-slots so each group covers more tokens.
```

The suggested microbatch is exact — 265 runs and 266 is refused. Pair a small slot count with a small
microbatch, or raise the slot count so each group covers more tokens.

Worth knowing if this limit is ever in the way: the per-group node allowance is deliberately generous, about
8.5x the nodes a group actually uses (measured 11.3 per group per layer against a budget of 96), and the
scheduler's own `max_splits = graph_size` assumption is what turns nodes into gigabytes. Tightening either
would move this limit a long way out.

### The saturation point applies to the group, not the microbatch

That bound saturates, and it saturates early. Once `n_expert_used × group` reaches `n_expert`, every expert
in the layer must be resident and paging buys nothing at all. So paging only has headroom while:

```
group  <  n_expert / n_expert_used        (group = --moe-chunk-size, default n_slots / n_expert_used)
```

| Model shape | Paging is useful only below |
| --- | --- |
| 128 experts, 8 used (e.g. Qwen3-235B-A22B, Qwen3-30B-A3B) | group = **16** |
| 256 experts, 8 used (e.g. Qwen3.6-35B-A3B) | group = **32** |

Grouping means this no longer caps the microbatch, but it does still cap how many tokens share an expert
matmul, and that is what sets prefill efficiency. Raising the slot count far enough to grow the group is the
same as not paging, so the ceiling is real — grouping moves where it binds, it does not remove it.

If a workload is prefill-heavy, measure prefill first and decide on that, not on generation throughput.

Larger slot counts raise the hit rate and cost more memory. The hit rate is reported at the end of a run.

## What it costs

The experts a token needs are read while the graph is running, so the compute waits on the read. Whether that
matters depends on where the bottleneck already is:

- **When the model does not fit in memory**, paging replaces demand paging by the OS - or an outright failure
  to load - with a bounded, explicit cache. This is the case it exists for.
- **When the model already fits**, paging is at best neutral and usually slower. Use full residency.
- **Combine it with `--load-mode none`.** Under the default `mmap` loading the whole file is mapped whether
  or not paging is on, so the pools only add memory on top of it - measurably worse on both axes.
- **On CPU specifically**, paging mostly reduces resident memory rather than improving speed. Compare it
  against ordinary `mmap` residency before assuming it helps; often the page cache is already doing a better
  job than a bounded pool can.
- **On a GPU**, experts are read into pinned host memory and copied into VRAM. That copy is serialised
  against compute in the current implementation, so throughput drops as the miss rate rises. A high hit rate
  is what makes it viable.

Paging deliberately conflicts with `--cpu-moe`, `--n-cpu-moe` and any `--override-tensor` that matches the
expert tensors: those decide where expert weights live at load time, while paging owns that decision itself.
Requesting both is an error rather than a silent precedence rule.

## Measured behaviour

Qwen3.6-35B-A3B-UD-IQ3_S (14.28 GiB, 41 layers, 256 experts, 8 used per token), CPU only, 6 threads,
microbatch 1, `--load-mode none`. Host: Linux, 23 GiB RAM, NVMe.

**Warm page cache**, `tg16`:

| Slots | Peak RSS | tg16 t/s | vs resident |
| ---: | ---: | ---: | ---: |
| resident | 14.81 GB | 9.62 | — |
| 8 | **2.62 GB** | 6.54 | −32 % |
| 16 | 3.02 GB | 7.36 | −24 % |
| 32 | 3.81 GB | 7.51 | −22 % |

**Cold page cache** (the model file evicted with `scripts/evict-file-cache.py` before each run):

| Slots | Peak RSS | tg16 t/s | Hit rate | Bytes read | Read time | Read bandwidth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resident | 14.81 GB | 9.13 | — | — | — | — |
| 8 | 2.62 GB | 5.10 | 26.1 % | 4841 MiB | 1444 ms | 3352 MiB/s |
| 32 | 3.81 GB | 5.15 | 56.1 % | 2881 MiB | 1405 ms | 2051 MiB/s |

So on this host paging buys a **5.6x smaller footprint for roughly a third of the throughput warm, and
somewhat under half cold**.

Those are 16-token runs, which are dominated by the compulsory misses of a cold pool. Longer runs amortise
them and land considerably better — the same cold 32-slot configuration over 128 tokens:

| Tokens | Hit rate | tg t/s cold | vs resident |
| ---: | ---: | ---: | ---: |
| 16 | 56.1 % | 5.07 | −44 % |
| 128 | **72.2 %** | **6.71** | **−26 %** |

Quote the longer figure when describing steady-state behaviour; the short one measures pool warm-up.

Three things in that data are worth taking seriously:

- **Warm numbers flatter the feature badly.** The warm run reads experts at 9.6 GiB/s, which is page cache,
  not storage. Any measurement taken shortly after downloading or hashing a model is a warm measurement.
  Always evict first.
- **More slots is not reliably faster.** Going from 8 to 32 slots more than doubles the hit rate (26 % to
  56 %) and reads 40 % fewer bytes, yet cold throughput is unchanged, because what remains is more scattered
  and the read time barely moves. Slot count buys memory predictability more than speed.
### Metal (Apple unified memory)

Validated on an M5 Pro (25.8 GB) with Qwen3-30B-A3B-Q6_K, `-ngl 99`, `llama-bench -p 32 -n 16 -r 3`. Cold
runs evicted the file by streaming 49.6 GiB of unrelated model files and confirming the reads reached the
SSD through `disk0` counters, since `purge` needs root.

| Slots | ub | Prefill warm / cold | Generation warm | Hit rate (tg) | Peak bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 10.02 / 10.18 | 9.27 | 29.1 % | 3.24 GB |
| 16 | 1 | 10.32 / 10.10 | 8.81 | 54.1 % | 4.73 GB |
| 32 | 1 | 10.63 / 10.61 | 9.09 | 69.4 % | 7.70 GB |
| 8 | 64 | 12.47 / 12.13 | 9.54 | 29.1 % | 3.31 GB |
| 16 | 64 | 13.93 / 13.76 | 9.22 | 54.1 % | 4.75 GB |
| 32 | 64 | 16.38 / 15.76 | 9.05 | 69.4 % | 7.71 GB |

Two things differ from the CPU picture and are worth knowing:

- **Cold and warm are within noise of each other**, and every run reported zero swaps. On this hardware
  storage is fast enough that reads are not the limiting factor, so byte-count improvements matter much less
  here than they do on a slower disk. Do not assume a policy change that reads fewer bytes will show up as
  throughput on a Mac.
- **The slot pools are not in the backend's printed Metal subtotal** — that reports only model, KV and
  compute. They are real, and they appear in the process peak; the memory breakdown accounts for them
  separately.

- **Paging never runs on top of `mmap`, because the combination does not work.** The mmap path registers the
  span covering a context's tensors as one backend buffer, and non-expert tensors are scattered through the
  file, so that span is effectively the whole model - the pools then sit on top of it. On CPU that measured
  as *higher* peak RSS than not paging at all (15.6 GB at 8 slots against 15.1 GB resident). On Metal it is
  fatal: a 151 GiB model was handed to the device as a 155 GiB mapped resource against an ~18 GiB working
  set and failed at the first decode with an out-of-memory command buffer, while 81 % of host memory was
  free. So `--load-mode auto` loads without mmap when paging is on, and an explicit `mmap` or `mmap+mlock`
  is refused. Paging wants `--load-mode none`, where the expert bytes genuinely never enter the process.

### Where the remaining cost is, and where it is not

The refill runs as a host graph node, which the GPU backends do not implement, so the scheduler splits the
graph around it. That sounds expensive and mostly is not:

- **Splits only grow when the paged layer is entirely GPU-resident.** Measured on CUDA: a fully offloaded
  model goes from 2 splits to 6 with two paged layers, i.e. **+2 per paged layer**. But with partial offload —
  the situation you are actually in when a model does not fit — the graph is already split-heavy and paging
  changes nothing: 62 splits with paging off, 62 with two layers paged, 62 with all layers paged.
- **The overhead is about 22.8 µs per paged layer per token**, measured on a fully offloaded model where
  reads are trivial (2413.6 → 2174.6 tok/s with two paged layers). For a 48-layer model at 10 tok/s that is
  ~1.1 ms of a 100 ms token, near **1 %** — and even that figure includes the admission and read work itself,
  not just the split.
- **Read concurrency is not the lever either.** 4, 16 and 32 read threads all give the same cold bandwidth
  (2062 / 2062 / 2045 MiB/s), because queue depth is bounded by the number of misses in a layer, not by the
  number of threads.

What is left is the disk reads themselves, which are inherently serialised against the compute that needs
them. Making paging materially faster means reading *less* (a higher hit rate) or reading *earlier*
(overlapping a layer's reads with compute that does not depend on them) — not removing graph splits.

### Optimisations considered, with their measured basis

Recorded so the analysis does not have to be redone. Each is marked done, rejected or still open.

**Rejected: a device-side handshake.** Replace the host refill node with an in-place device stall — a kernel
publishes the routed ids to shared memory, the device stalls on an event (`MTLSharedEvent` on Metal,
`cuStreamWaitValue32` on CUDA), and a CPU sidecar reads the experts and releases it. This is what the original
proof of concept did. It was measured and rejected: it removes the round trip but **not the wait**, since the
device still stalls for the duration of the reads. It targets the ~1 % above while the real gap is ~26 %, and
it costs a documented deadlock risk — NVIDIA's own documentation warns "improper use of this API may deadlock
the application". Worth revisiting only if the split cost is ever shown to matter on a given machine.

*Caveat if it is revisited:* on Apple unified memory `-ngl 99` with paging **is** full GPU residency, which is
the one configuration where splits do grow (+2 per paged layer, ~96 for a 48-layer model). Measure there
before assuming the CUDA result carries over.

**Done: read straight into unified memory** — and it was the single largest win in the feature's history.
Validated on an M5 Pro, like-for-like at 8 slots and microbatch 1: generation went from **6.6 to 11.2 t/s,
+69.7 %**, at an unchanged 3.29 GB footprint and with correctness confirmed bit-exact beforehand. That also
overtakes the original proof of concept's 10.1 t/s by 10.9 %, closing a gap that had been open since the
rewrite. Experts used to be read into a staging buffer and handed over
with `ggml_backend_tensor_set` on every backend whose buffers are not host-addressable. Metal reports
`ggml_backend_buffer_is_host() == false` for all of its buffer types even when the memory is genuinely
unified, so it took that path and paid a full second copy of every expert byte. Backends can now answer a
`ggml_backend_buffer_is_host_writable` query instead; Metal answers it from
`ggml_metal_buffer_is_shared`, and anything without the entry point falls back to the generic answer, so CPU
and CUDA are unaffected. Measured motivation: on an M5 Pro, prefill moved 124 GiB at 15.6 GiB/s — RAM speed,
i.e. the "reads" were cache hits being copied twice.

**Rejected after building it: overlap a layer's reads with compute that does not depend on them.** The down
projection is not consumed until the gate/up matmuls have run and is 45 % of the bytes an expert occupies, so
the refill was split — read gate/up, leave the down read in flight, collect it just before the matmul that
consumes it. It was implemented, verified byte-identical on CPU, CUDA and Metal, measured, and removed.

The mechanism worked and was still a loss on every backend:

| | prefill | generation | graph splits |
| --- | ---: | ---: | ---: |
| CUDA, 32 slots | −16.9 % | −10.6 % | 82 → 162 |
| Metal, 16 slots | −13.9 % | −15.3 % | 98 → 194 |
| Metal, 32 slots | −17.1 % | −22.1 % | 98 → 194 |
| CPU, 32 slots | +0.6 % (noise) | — | 1 → 1 |

Collecting the read needs a host-side node between two device matmuls, and the scheduler splits the graph
around every one — exactly **two extra splits per paged layer**. Each split is a device synchronisation, and
that costs more than the overlap hides. Read time did fall, so the reads genuinely finished sooner; the wall
clock still got worse.

**The premise was sound, which is why this is worth recording rather than just deleting.** A clean
measurement on Metal (`-r 1 --no-warmup`, so the read statistics and the throughput denominator cover the
same tokens) puts reads at **51-59 % of wall time** at microbatch 1 — there really was about half the run
available to hide behind. What sank it is that any host-side wait between two device matmuls costs a split,
and the split is worth more than the read. The only way to wait without one is a device-side wait, which is
the handshake rejected above for its deadlock risk. Earlier figures of 82 % came from a warmup-contaminated
shape and should not be reused.

On CPU it was a wash for a different reason: the reader threads and the compute threads are the same cores,
so there is no second execution resource for the read to overlap with.

**Done: chunk the microbatch inside the MoE layer** — shipped as token groups; see *Token groups* above for
what it delivered (**+9.1 %** prefill at microbatch 512, +6.9 % at 64, against the old ceiling of 4). The
analysis below is kept because it predicted the outcome correctly, including the modest size of it. The slot
bound existed because one
`mul_mat_id` consumes every token in the ubatch at once, so every expert any of them routed to has to be
resident together. But the MoE FFN is per-token: splitting the ubatch into sub-groups and running the expert
matmul once per sub-group is mathematically identical. That would decouple the slot count from the global
microbatch, letting attention and the dense path run at `--ubatch-size 512` while the MoE layer internally
works in groups of, say, 8 — turning the saturation limit above from a hard wall into a tunable.

It is the only idea here that addresses the prefill problem rather than the read cost, but **its ceiling is
lower than it first appears**. Chunking to groups of C gives the MoE path exactly the same read amortisation
as simply running a global microbatch of C — the expert matmul sees C tokens either way. The only thing it
actually buys is letting *attention and the dense path* run at the full microbatch. So the gain is bounded by
how much those layers care about batch size, not by how much the MoE layer does.

**On CPU it is not worth it; on GPU it is.** Qwen3.6-35B-A3B, `pp256`, paging off: raising the microbatch
from 4 to 512 is worth **+20 %** on CPU (21.93 → 26.39) but **+141 %** on CUDA (27.12 → 65.46). Prefill on a
GPU is far more batch-sensitive, and the GPU is where a model this size actually runs.

The isolated measurement is the convincing one. Holding the slot count fixed at 32 — so the residency policy
is *identical*, hit rate 73.21/73.22/73.23 % and bytes read within 0.04 % — and varying only the microbatch,
full GPU residency:

| Microbatch | `pp256` | vs microbatch 1 |
| ---: | ---: | ---: |
| 1 | 26.95 | — |
| 2 | 32.20 | +19.5 % |
| 4 | 36.67 | +36.1 % |

Same reads, same evictions, **+36 % prefill purely from batching**, and 4 is as far as 32 slots allow. That
is exactly the ceiling chunking would lift, and the no-paging figures above show substantial headroom past 4.

A useful signal that prefill is being crippled: on Metal at microbatch 1, `pp` and `tg` were 7.29 and 6.18
tok/s, barely apart, where a healthy configuration has prefill several times generation. Both halves of that
have since been addressed — direct reads lifted generation and grouping lifted prefill — so the symptom is
worth remembering as a diagnostic rather than as a current result.

Also note the trade: smaller groups amortise each expert read across fewer tokens, and a group boundary can
evict what the next group needs, so bytes read would rise.

**Done, but only half of it worked: read fewer bytes.** Eviction now goes by frequency rather than recency
(see *How it works*). Replaying routing traces offline, that reads 12 % fewer bytes at 32 slots over 128
tokens and 20 % over 453, against a Belady bound of 38-44 %. The implementation reproduces the replay
exactly — 10091 misses predicted, 10091 measured.

**It does not make anything faster here, and the reason is worth recording.** Cold, 32 slots, 128 tokens:

| | LRU | LFU |
| --- | ---: | ---: |
| Bytes read | 13751 MiB | 12098 MiB (−12.0 %) |
| Read time | 3779 ms | 3670 ms (−2.9 %) |
| Effective read bandwidth | 3639 MiB/s | 3297 MiB/s |
| `tg128` | 6.73 | 6.76 |

Twelve percent fewer bytes bought three percent less read time, because the bytes that remain are worse:
keeping the popular experts resident means the misses that are left are the rare, scattered ones, and
effective bandwidth falls accordingly. Throughput is unchanged within run-to-run variance. Warm is likewise
a wash (7.87 ± 0.23 against 8.02 ± 0.08).

So the byte reduction is real, repeatable and deterministic, and the throughput gain is not there on an NVMe
host. Keep it for the I/O reduction and do not quote a throughput number for this hardware.

**On Metal it does convert to throughput**, which is the case that matters, and it generalises to a different
routing shape — an M5 Pro running Qwen3-30B-A3B, 128 experts with 8 used, against the 256/8 above:

| Slots | PP bytes | TG hit rate | Warm PP | Warm TG |
| ---: | ---: | ---: | ---: | ---: |
| 8 | −10.8 % | +5.80 pp | +4.7 % | +3.2 % |
| 16 | −12.1 % | +2.13 pp | +12.3 % | +6.1 % |
| 32 | −12.1 % | +1.63 pp | **+29.9 %** | +19.3 % |

The byte reduction reproduces the figure measured here almost exactly; what differs is that reads are around
half of Metal's wall clock rather than a fifth, so fewer bytes becomes less time. Treat the byte column as
solid and the throughput columns as directional — they are short `-p 32 -n 16 -r 3` runs.

**The counters do not need to decay.** `use_count` never decays, so an expert that was popular early could in
principle hold a slot forever. Over 4096 generated tokens on Metal the hit rate instead *rose* — 68.10 % to
74.54 % at 16 slots, 82.68 % to 89.70 % at 32 — with no swapping. Note this comparison is against the same
configuration's 128-token run, which conflates pool warm-up with policy, so the claim it supports is the
narrow one: nothing rots over a long session. Periodic halving of the counters was also tried offline and
made no difference, so plain LFU ships without a tuning knob.

Two ideas from this line are still untried: refusing admission to experts routed only once in a ubatch, and
carrying residency across the requests of one conversation.

**Not a lever: read concurrency.** Already measured — 4, 16 and 32 read threads give identical cold
bandwidth, because queue depth is bounded by the number of misses in a layer, not by the number of threads.
Raising `--moe-read-threads` will not help.

## How it works

Expert weight matrices for paged layers are given metadata but no storage - no buffer is allocated for them
and their bytes are never read at load time. Each is replaced in the graph by a pool holding `n_slots`
experts, allocated on whatever buffer type the rest of the layer landed on.

The routed expert ids only exist once the router has run, so a node between the router and the expert
matmuls turns expert ids into slot indices, reading in whatever is missing on the way. Eviction takes the
least frequently routed expert, with recency as the tiebreak — routing is skewed enough that how often an
expert is used predicts reuse better than how recently. Slots touched by the current ubatch are pinned so an
expert admitted early cannot be evicted by a later token in the same ubatch.

The policy cannot affect results. It chooses which slot holds an expert, never which expert is used, so the
equivalence tests hold whatever it decides.

Only the expert matmuls are indexed by slot. Routing weights, per-expert biases, per-expert scales and LoRA
weights are all still sized by `n_expert` and keep indexing by expert id.

Reads are positional and bounds-checked against the file length. A pool that cannot be allocated, an
out-of-range expert id, a truncated file or a failed read all fail the decode with a propagated error - never
an abort, never a silent fallback to a different slot count or backend, and never a partially filled slot
presented as valid.

## Statistics

`llama_moe_stats()` reports lookups, hits, misses, evictions, reads, bytes read and read time, summed over
every paged layer. Tools that print a performance summary print these alongside it, once per run.

A low hit rate means the slot count is too small for the routing pattern: either raise it, or accept that the
model's routing is too diffuse for paging to help.

### Routing traces

`LLAMA_MOE_TRACE=<path>` writes one line per resolve — `<layer> <n> <id> <id> ...` — recording what the
router asked for. Routing depends only on the model and the input, not on the slot count or the admission
policy, so a single capture can be replayed offline against any pool size or candidate policy. That makes
"would a different policy read fewer bytes?" answerable without a rebuild-and-benchmark cycle per candidate,
and a replay of the current policy can be checked against the counters the run itself reports.

Tracing is off unless the variable is set, and costs one null check per resolve when it is.

## Testing

- `test-moe-residency` - the admission policy and the read path, no model required.
- `test-moe-paging` - loads a generated MoE model paged and fully resident, decodes the same tokens through
  both and requires the logits to match exactly. Registered twice, once on CPU and once with the layers
  offloaded, so whichever GPU backend is present is covered.
