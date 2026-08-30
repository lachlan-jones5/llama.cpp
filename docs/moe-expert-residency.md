# MoE expert residency (expert paging)

Mixture-of-Experts models carry a lot of weight that any single token does not touch. A layer may hold 256
experts and route each token to 8 of them. Expert paging keeps only a bounded number of experts of each layer
resident and reads the rest from the GGUF as the router selects them.

This trades throughput for memory. It is off by default and only worth turning on in one situation: **the
model does not fit, but its working set does.**

## Options

| Option | Meaning |
| --- | --- |
| `--moe-n-slots N` | Keep `N` experts of each paged layer resident. `0` (default) disables paging entirely. |
| `--moe-n-layers N` | Page only the first `N` MoE layers. `0` (default) pages every MoE layer. |
| `--moe-read-threads N` | Threads used to read experts. `0` (default) picks a sensible number. |

Each has a matching `LLAMA_ARG_*` environment variable.

### Choosing a slot count

A ubatch of `N` tokens can route to `n_expert_used * N` distinct experts, and every one of them has to be
resident at the same time. So:

```
n_slots >= min(n_expert, n_expert_used * n_ubatch)
```

A ubatch is not bounded by `n_ubatch` alone. With several sequences in flight the batch allocator packs a
token from each active sequence into one ubatch, so the requirement scales with `--parallel` as well:

```
n_slots >= min(n_expert, n_expert_used * n_ubatch * n_parallel)
```

This is checked before inference and refused if it cannot be satisfied, naming the value that would fix it.
The practical consequence is that **small slot counts require a small microbatch and few parallel
sequences**. For a model using 8 experts per token, `--moe-n-slots 8` requires `--ubatch-size 1` and
`--parallel 1`; the same model with `--parallel 4` needs 32 slots, and at the default `--ubatch-size 512` it
needs 256 slots, which is most of the layer and defeats the purpose.

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
- **Paging on top of `mmap` is a pure loss.** With the default mmap loading, the same sweep produced *higher*
  peak RSS than full residency (15.6 GB at 8 slots against 15.1 GB resident) and lower throughput: the whole
  file is mapped either way and the pools are simply added on top. Expert paging is worth using with
  `--load-mode none`, where the expert bytes genuinely never enter the process.

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

Recorded so the analysis does not have to be redone. None of these are implemented.

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

**Done: read straight into unified memory.** Experts used to be read into a staging buffer and handed over
with `ggml_backend_tensor_set` on every backend whose buffers are not host-addressable. Metal reports
`ggml_backend_buffer_is_host() == false` for all of its buffer types even when the memory is genuinely
unified, so it took that path and paid a full second copy of every expert byte. Backends can now answer a
`ggml_backend_buffer_is_host_writable` query instead; Metal answers it from
`ggml_metal_buffer_is_shared`, and anything without the entry point falls back to the generic answer, so CPU
and CUDA are unaffected. Measured motivation: on an M5 Pro, prefill moved 124 GiB at 15.6 GiB/s — RAM speed,
i.e. the "reads" were cache hits being copied twice.

**Candidate: overlap a layer's reads with compute that does not depend on them.** Today one refill fetches
the gate, up and down pools together and everything waits. The down projection is not consumed until after
the gate/up matmuls have run, so the refill could be split: fill gate/up, let compute proceed, and fetch down
concurrently. That hides roughly a third of the read stall, needs no device stall, and leaves the admission
policy and the correctness tests untouched. This is the most promising of the three.

**Candidate: read fewer bytes.** At 32 slots over 128 cold tokens the hit rate is 72.2 %, so there is
headroom, and every avoided miss is ~400 KiB not read. Options: pin a per-layer hot set that is never
evicted, refuse admission to experts routed only once in a ubatch, or carry residency across the requests of
one conversation. Payoff is uncertain — the routing may simply be too diffuse.

**Not a lever: read concurrency.** Already measured — 4, 16 and 32 read threads give identical cold
bandwidth, because queue depth is bounded by the number of misses in a layer, not by the number of threads.
Raising `--moe-read-threads` will not help.

## How it works

Expert weight matrices for paged layers are given metadata but no storage - no buffer is allocated for them
and their bytes are never read at load time. Each is replaced in the graph by a pool holding `n_slots`
experts, allocated on whatever buffer type the rest of the layer landed on.

The routed expert ids only exist once the router has run, so a node between the router and the expert
matmuls turns expert ids into slot indices, reading in whatever is missing on the way. Admission is LRU, and
slots touched by the current ubatch are pinned so an expert admitted early cannot be evicted by a later token
in the same ubatch.

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

## Testing

- `test-moe-residency` - the admission policy and the read path, no model required.
- `test-moe-paging` - loads a generated MoE model paged and fully resident, decodes the same tokens through
  both and requires the logits to match exactly. Registered twice, once on CPU and once with the layers
  offloaded, so whichever GPU backend is present is covered.
