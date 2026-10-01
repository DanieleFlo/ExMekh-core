# Indice: upstream

Indice generato; non modificare a mano.

## [Coding Agent Clients](CLIENTS.md)

Add this provider to `~/.pi/agent/models.json`:

- righe 16–58: **Pi** — Add this provider to `~/.pi/agent/models.json`:
- righe 59–86: **OpenCode** — Merge the provider into `~/.config/opencode/opencode.json`:
- righe 87–102: **Codex CLI** — Use the Responses API.
- righe 103–125: **Claude Code** — Use the Anthropic-compatible endpoint.

## [NVIDIA CUDA GPUs](CUDA_MULTI_GPU.md)

Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.

- righe 11–17: **Build** — Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.
- righe 18–90: **For an Ada build:** — make cuda CUDA_ARCH=sm_89
- righe 24–38: **Normal layer placement** — Without tensor parallelism, DwarfStar places complete layers on the selected
- righe 39–69: **Flash tensor parallelism** — `--cuda-tensor-parallel` pairs GPUs and divides routed-expert work inside
- righe 70–90: **Serve multiple users** — ./ds4-server --cuda --cuda-tensor-parallel \

## [DGX Spark](DGX_SPARK.md)

make cuda-spark

- righe 9–21: **Build and run** — make cuda-spark
- righe 22–62: **DeepSeek V4.1 Flash** — Q2 needs SSD streaming on one 128 GB Spark:
- righe 63–83: **Qwen3.8 Flash Next** — Both Q2 and Q4 fit on one Spark.
- righe 84–95: **GLM 5.3 Flash** — Q2 is the resident target for one Spark:
- righe 96–111: **Vision and speculative decoding** — DeepSeek Vision Experimental needs its matching text model and encoder:
- righe 112–140: **Flash Q2 with DSpark** — September 6, 2026, fully resident Flash 0731: median generation speed from
- righe 141–150: **Larger models and serving** — CUDA also has SSD streaming paths for larger weights.

## [Inference Across Machines](DISTRIBUTED.md)

This is a 50/50 split with exactly one worker.

- righe 17–42: **Tensor parallelism between two Macs** — This is a 50/50 split with exactly one worker.
  - righe 29–42: **Link setup** — Use a Thunderbolt cable.
- righe 43–44: **Machine A, example member interface en1.** — sudo ifconfig en1 inet 10.99.0.2/30 alias
- righe 45–70: **Machine B, example member interface en6.** — sudo ifconfig en6 inet 10.99.0.1/30 alias
  - righe 60–70: **Start the pair** — Download the same model on both machines.
- righe 71–74: **Machine B.** — ./ds4 --tensor-parallel --role worker \
- righe 75–114: **Machine A.** — ./ds4 --tensor-parallel --role coordinator \
- righe 100–114: **Tensor parallelism between two Sparks** — V4.1 Flash Q2 text inference supports one GPU per rank, with a 50/50 expert
- righe 115–119: **Worker.** — ./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
- righe 120–146: **Coordinator.** — ./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 32768 \
- righe 137–146: **Pipeline parallelism** — Each process maps only its assigned layers, retaining that slice of the KV
- righe 147–149: **Machine A.** — ./ds4 --role coordinator --layers 0:19 --listen 10.99.0.2 9911
- righe 150–168: **Machine B.** — ./ds4 --role worker --layers 20:output --coordinator 10.99.0.2 9911
  - righe 164–168: **Full PRO Q4** — For two 512 GB Mac Studios, use the split artifacts:
- righe 169–173: **Machine A.** — ./download_model.sh pro-q4-layers00-30
- righe 174–199: **Machine B.** — ./download_model.sh pro-q4-layers31-output
  - righe 183–199: **Tuning and recovery** — Keep default chunk sizes first.

## [Metal on Apple Silicon](METAL.md)

Install Apple's command-line developer tools if needed:

- righe 5–24: **Build and run** — Install Apple's command-line developer tools if needed:
- righe 25–55: **Choose a model** — Here, Flash means DeepSeek V4 Flash.
- righe 56–62: **Two Macs** — Two 128 GB Macs can run a larger model fully resident with a 50/50 routed-expert
- righe 63–71: **Next steps** — For DeepSeek V4 Flash and PRO, `--power 70` trades throughput for lower

## [Models and Vision](MODELS.md)

./download_model.sh ds4f-q2

- righe 17–45: **DeepSeek V4** — ./download_model.sh ds4f-q2
- righe 46–119: **DeepSeek V4.1 Flash** — V4.1 Flash text and vision inference work on Metal.
- righe 120–146: **Qwen3.8 Flash Next** — `./download_model.sh qwen38-q2` downloads one **137.10 GiB** GGUF: 41.73 GiB
- righe 147–176: **GLM 5.3 Flash** — ./download_model.sh glm53-q2
- righe 177–199: **Full GLM 5.3 and GLM 5.2** — Full GLM 5.3 Q2 is about 197 GiB.
- righe 200–250: **Vision** — PNG and JPEG input works in the CLI, native agent, and HTTP server on Metal,
  - righe 206–219: **DeepSeek Flash Vision Experimental** — Vision Experimental is a different language checkpoint from Flash 0731.
  - righe 220–237: **GLM 5.3 Flash** — The text GGUF stays the same.
  - righe 238–250: **Qwen3.8 Flash Next** — The text GGUF stays the same.

## [Performance and Benchmarking](PERFORMANCE.md)

`ds4-bench` measures prefill and generation at successive context frontiers:

- righe 10–34: **Context sweeps** — `ds4-bench` measures prefill and generation at successive context frontiers:
- righe 35–58: **Recorded Flash Q2 baseline** — These existing sweeps use 2048-token intervals and 128 generation tokens per
- righe 59–72: **What to compare next** — from a warm cache.

## [Qwen3.8 Flash Next](QWEN38_FLASH_NEXT.md)

make

- righe 9–71: **Download and run** — make
- righe 72–88: **DGX Spark performance** — Measured on a single Spark with resident weights and disk-only n-grams:
- righe 89–97: **Conversion** — See GGUF conversion for conversion from the HF
- righe 98–162: **Vision** — Images go through the model's Qwen3-VL vision tower.
- righe 163–257: **Validation** — make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse test-q8-prefill-variants

## [Serving Models](SERVER.md)

./ds4-server --ctx 32768

- righe 5–21: **Start locally** — ./ds4-server --ctx 32768
- righe 22–51: **APIs** — The Flash and PRO names accepted by the model endpoints are compatibility
- righe 52–93: **Multiple sessions** — ./ds4-server --ctx 4096 --batched-session 4
- righe 94–103: **Images** — Start with the matching language GGUF and `--vision FILE`; see
- righe 104–132: **Disk KV cache** — Disk caching saves useful prefixes across slot reuse and server restarts:
- righe 133–149: **Tool history and debugging** — For DeepSeek, the server preserves sampled DSML tool blocks and assigns

## [Speculative Decoding](SPECULATIVE_DECODING.md)

DSpark is a separate support GGUF, not a standalone language model.

- righe 13–47: **DeepSeek Flash: DSpark** — DSpark is a separate support GGUF, not a standalone language model.
- righe 48–59: **GLM: built-in MTP** — GLM's draft block is already in its main GGUF:
- righe 60–79: **Qwen3.8: built-in MTP** — Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:
- righe 80–107: **Sampling and reproducibility** — At temperature zero, accepted drafts must match the target's greedy

## [Models Larger Than RAM](SSD_STREAMING.md)

./ds4 -m ds4flash.gguf --ssd-streaming

- righe 14–26: **Start with the automatic budget** — ./ds4 -m ds4flash.gguf --ssd-streaming
- righe 27–30: **Flash Q2 on a smaller Mac.** — ./download_model.sh ds4f-q2
- righe 31–34: **GLM 5.3 Flash Q4 on a 128 GB Mac.** — ./download_model.sh glm53-q4
- righe 35–98: **PRO Q2 on a 128 GB Mac: usable for inspection, but slow.** — ./download_model.sh pro-q2-imatrix
- righe 44–75: **Adjusting the cache** — To leave more room for context or other sessions:
- righe 76–98: **M5 Max measurements** — On a 128 GB M5 Max, with automatic cache sizing and no speculative decoding:

## [AMD Strix Halo](STRIX_HALO.md)

For a container setup, see the maintained

- righe 9–33: **Prerequisites** — For a container setup, see the maintained
- righe 34–51: **GPU-visible memory** — Check the memory pool reported by `rocminfo`.
- righe 52–64: **Build and run Flash** — make strix-halo
- righe 65–85: **GLM 5.3 Flash** — The reference Q2 setup uses SSD streaming to leave room for its graph and KV

## [Testing and Development](TESTING.md)

Model-free checks include:

- righe 10–37: **Focused checks** — Model-free checks include:
- righe 38–51: **Investigating output** — ./ds4 --dump-tokens -p "..."
- righe 52–66: **Tools and source references** — calibration corpus
