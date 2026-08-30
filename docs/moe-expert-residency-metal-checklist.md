# Metal acceptance checklist for MoE expert residency

Everything in this checklist has to be run on Apple hardware. It has **not** been run — the development host
is Linux and cannot validate the Metal backend. Nothing below should be reported as passing until it has
actually been executed on the Mac.

Branch: `moe-expert-residency-portable`, based on `official/master`.

## What to expect

The implementation is backend-independent. Expert reads go into a staging buffer and are handed to the
backend with `ggml_backend_tensor_set`, which for a Metal shared (unified memory) buffer is a plain `memcpy`.
So Metal is expected to work **without any Metal-specific code**. If it does not, that is a real finding.

There is no Metal interceptor kernel and no `MTLSharedEvent` handshake in this branch. Those belong to the
device-side overlap optimisation, which is not implemented yet. Expect reads to be serialised against
compute.

## 1. Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_FATAL_WARNINGS=ON
cmake --build build --config Release -j $(sysctl -n hw.logicalcpu)
```

Build the non-embedded shader path too, since upstream CI only exercises one of them. Note it needs Xcode's
optional Metal toolchain selected explicitly — `xcrun -sdk macosx metal` will not find it otherwise:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
TOOLCHAINS=<com.apple.dt.toolchain.Metal.*> \
cmake -B build-noembed -DCMAKE_BUILD_TYPE=Release -DGGML_METAL_EMBED_LIBRARY=OFF -DGGML_METAL_SHADER_DEBUG=ON
cmake --build build-noembed -j $(sysctl -n hw.logicalcpu)
```

- [ ] both configurations build
- [ ] no **new** warnings in files this branch touches

> Do not treat the non-embedded path as a gate on this work. It has a pre-existing upstream defect,
> independent of expert paging: the build-time `metal` invocation passes no feature macros, so the prebuilt
> `default.metallib` contains no BF16 kernels, while the runtime advertises `has_bfloat` on capable hardware
> and asks for them anyway. On an M5 Pro this makes `test-backend-ops -b MTL0 -o MUL_MAT_ID` abort with
> *"Function kernel_mul_mv_id_bf16_f32_4 was not found in the library"*. The MoE tests pass on that build —
> the missing kernels are BF16 variants a Q6_K model never uses. Likewise `upscale.metal`'s unused
> `bilinear_tri` warning is upstream and untouched by this branch.

## 2. Unit and equivalence tests

```sh
cd build
ctest -L main -E "test-llama-archs" --output-on-failure
```

- [ ] `test-moe-residency` passes (policy and read path, no model)
- [ ] `test-moe-paging` passes (CPU placement)
- [ ] `test-moe-paging-gpu` passes — this is the one that exercises **Metal**, since it offloads the layers

`test-moe-paging-gpu` requires the paged logits to match the fully resident logits exactly (nmse `0.0`), and
requires the run to have taken real misses and evictions, so a pass means paging genuinely happened.

## 3. Backend operation tests

The Metal device is named `MTL0` (`ggml-metal-device.m` builds it as `"MTL" + index`).

```sh
./bin/test-backend-ops test -b MTL0 -o MUL_MAT_ID
./bin/test-backend-ops test -b MTL0 -o GET_ROWS
./bin/test-backend-ops test -b MTL0 -o ARGSORT
./bin/test-backend-ops test -b MTL0 -o SOFT_MAX
```

- [ ] all four report OK on the **embedded** build (the reference run on CUDA0 gave 872/872 for `MUL_MAT_ID`)

On the non-embedded build `MUL_MAT_ID` is expected to abort on the first BF16 case for the upstream reason
above. That is not a result about this branch; record it and move on.

## 4. Real model

Model: `Qwen/Qwen3-30B-A3B-GGUF`, file `Qwen3-30B-A3B-Q6_K.gguf`
Size `25092531712`, SHA-256 `0c1754eaf4514d9cb6aaf1a6a2c15cd496d56b8413843b2ea9328f215a5eb8ff`.

Verify the hash before trusting any numbers from it.

This model uses 8 experts per token, so **8 slots requires `--ubatch-size 1`**. At a larger microbatch the
configuration check will refuse it and tell you the slot count it needs.

### 4a. Deterministic equivalence

Run once fully resident and once paged, greedy, and require identical output:

```sh
for slots in 0 8 16 32; do
  ./bin/llama-cli --model /path/to/Qwen3-30B-A3B-Q6_K.gguf \
    ${slots:+--moe-n-slots $slots} \
    --ubatch-size 1 --batch-size 8 --ctx-size 1024 \
    --n-gpu-layers 99 --no-mmap --no-warmup --fit off \
    --temp 0 --single-turn --simple-io --no-display-prompt \
    --predict 128 --prompt 'In exactly 120 words, explain how an LRU cache works.'
done
```

- [ ] 8, 16 and 32 slots each produce output identical to the fully resident run
- [ ] the arithmetic prompt used previously still returns exactly `703` at all three slot counts
- [ ] no run silently falls back to full residency (the memory figures below should differ)

### 4b. Measurements

```sh
./bin/llama-bench -m /path/to/Qwen3-30B-A3B-Q6_K.gguf \
  -ngl 99 -p 32 -n 16 -r 3 -ub 1 -b 8 --moe-n-slots <0|8|16|32>
```

Record for each slot count:

| Slots | Peak footprint | Metal allocation | tg t/s | pp t/s | Hit rate | Bytes read | Read time | Swap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (resident) | | | | | n/a | n/a | n/a | |
| 8 | | | | | | | | |
| 16 | | | | | | | | |
| 32 | | | | | | | | |

The hit/miss/eviction/bytes/read-time figures are printed by `llama-bench` at the end of each run.

Earlier proof-of-concept figures on an M5 Pro were 3.27 GB at 10.1 tok/s (8 slots), 4.75 GB at 10.5 tok/s
(16), 7.70 GB at 12.9 tok/s (32). **Treat those as unvalidated**: they were measured immediately after the
model was downloaded and hashed, so the page cache was almost certainly warm.

### 4c. Cold cache

The numbers above are meaningless if the file is already in the page cache. On macOS use `purge`, or read a
large unrelated file to evict it, and confirm with `fs_usage` or `iostat` that reads actually reach the SSD.

- [ ] each slot count measured warm **and** cold, and labelled as such

## 5. Failure and lifecycle

- [ ] `--moe-n-slots 4` with 8 experts per token is refused before inference with a message naming the
      required count, not an abort
- [ ] `--moe-n-slots 8 --ubatch-size 4` is refused, naming 32 as the needed count
- [ ] `--moe-n-slots 8 --n-cpu-moe 4` is refused as a conflict
- [ ] truncating or replacing the model file mid-run surfaces a propagated error, not a crash
- [ ] `llama-cli` exits cleanly with no orphan processes
- [ ] `llama-server` starts, serves a request, and shuts down with no orphan processes
- [ ] concurrent requests and multiple server slots produce correct output
- [ ] prompt cache reuse and graph reuse produce correct output

## 6. Reporting

For each run record backend and exact device, model hash and quantisation, context/batch/microbatch/layers/
slots, host RSS and peak, Metal allocation, swap activity, hit/miss/eviction counts, bytes read, read latency
and bandwidth, prompt and generation throughput, time to first token, and whether the page cache was warm or
cold.

Do not change any production launcher configuration until this suite passes.
