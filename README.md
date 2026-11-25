# Strata-gfx906-Argon (Vega 20 `gfx906` + AVX2 Optimization Fork)

> **Upstream Base:** Strata `v0.2.20` (`engine 0.1.66`, commit `a34979f`)
> **Optimization Authors:** Gemini Argon and Neon
> **Target Hardware:** AMD Vega 20 (`gfx906`: Radeon Pro VII / Instinct MI50 / MI60, wave64) + x86_64 AVX2-only CPUs (e.g., Intel Xeon E5 v4 Broadwell-EP)
> **Quality Guarantee:** **Zero Quality Degradation / 100% Bitwise Parity** — verified with `20/20` CTest parity suites (`0 failures`) and exact FNV-64 output hashes.

---

## What Is Patched in This Fork

All changes are strictly scoped to **22 files under `engine/`** (`21` modified + `1` added header: `engine/include/strata/kernels/gfx906_dlops.hpp`) and preserve exact mathematical parity with upstream across `IQ3_XXS`, `IQ3_S`, `IQ4_XS / IQ4_NL`, `Q4_K / Q5_1 / Q8_0`, and `BF16` workloads.

### 1. `gfx906` ROCm HIP Compatibility (`engine/include/strata/platform/hip_compat/`)
- **`strata_hip.h` & `cublas_v2.h`:** Added missing CUDA-to-HIP symbol mappings required to compile `v0.2.20` with `-DSTRATA_HIP_GFX906=ON` (`cudaDeviceGetP2PAttribute`, `cudaDeviceP2PAttr`, `cudaDevP2PAttrNativeAtomicSupported`, `cudaDevP2PAttrPerformanceRank`, `cudaDevP2PAttrAccessSupported`, `cudaDevP2PAttrCudaArrayAccessSupported`, `cudaDeviceGetPCIBusId`, `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`) along with `gfx906` compatibility helpers.

### 2. `gfx906` GPU Decode, BF16 & MoE Kernel Optimizations (`engine/src/kernels/cuda/`)
- **`gfx906_dlops.hpp` (new) & `s26_tsum.cuh`:** Wave64 / DLOPS primitives (`v_dot4_i32_i8`, `v_perm_b32`, `DualScaledHalf2` `v_dot2_f32_f16`) and zero-address cross-lane reductions combining **`ds_swizzle_b32`** BitMode (`0x401F` for XOR-16, `0x101F` for XOR-4) with 1-cycle **GFX9 DPP** (`row_ror:8`, `quad_perm:[2,3,0,1]`, `quad_perm:[1,0,3,2]`).
- **`native_bf16.cu`:** Added explicit `__launch_bounds__(BLOCK_SIZE, ...)` to `bf16_f32_mmvf_multi_kernel` and `bf16_f32_mmvf_rows_kernel` (eliminating **49 VGPR spills** / `120 buffer_load` + `122 buffer_store` scratch instructions caused by HIP's default 1024-thread assumption -> **0 spills across the entire TU**), and replaced all **5,010 `ds_bpermute_b32`** instructions with **`ds_swizzle_b32`** (`ds_bpermute_b32: 5010 -> 0`).
- **`iq_kernels.cu` & `native_mmvq.cu`:**
  - Replaced emulated CUDA `__vcmpne4` / `__vsub4` and 2 KB LDS sign-mask lookups in `Split<18>` (`IQ3_XXS`), `Split<21>` (`IQ3_S`), and `DotG` with a 3-instruction ALU sign helper (`gfx906_sign_nz` / `nib_sign_nonzero_grid` via `__umul24` + XOR + ADD).
  - Replaced memory table loads of `kvalues_iq4nl` with SGPR immediate constants + native `v_perm_b32` (`__builtin_amdgcn_perm`) in `get_int_from_table_16_iq4nl` and `iq4_table_lookup`.
  - Replaced `~17,800` `ds_bpermute_b32` instructions across `warp_sum`, `row_dot2_80_sub16`, `row_dot_40_sub8` (`IQ4_NL` down), `row_dot_{20,24}_sub8`, and fused SwiGLU+`q8_1` epilogues with `ds_swizzle_b32` + interleaved DPP.
  - Fixed the `native_gu_fused_kernel<21>` (`IQ3_S`, `STRATA_EXP_MODE=8`) register cliff: trimmed 2 KB of fixed LDS (`5120 B -> 3072 B`), added `#pragma nounroll` on row-pass loops, and enforced `__launch_bounds__(256, 2)` for core quant formats (`18, 20, 21, 23, 42` gate/up; `12, 18, 20, 21, 23, 42` down). Drops `IQ3_S` gate/up from **195 -> 128 VGPRs** (doubling CU occupancy from 1 to 2 blocks/CU) and keeps `IQ4_NL` (`116 VGPR`), `IQ4_XS` (`121–127 VGPR`), and `Q4_K` down (`125 VGPR`) at **2 blocks/CU with 0 spills**.
  - Enabled `g_stage_grid_mmvq` and `kExpModeDefault = 8` on `gfx906`.
- **`gdn.cu`, `native_gdn.cu`, `fused_gr.cu`, `bf16_gemv.cu`, `qsa_decode_attn.cu`, `verify_kernels.cu`:** Targeted `gfx906` wave64 and DLOPS optimizations across GDN recurrence, BF16 GEMV, QSA decode attention, and verify kernels.

### 3. `gfx906` Prefill `int8` & GEMM Optimizations (`engine/src/kernels/cuda/`, `engine/src/prefill/`)
- **`i8_expert.cuh` & `i8h_volta.cuh`:** Replaced 8x emulated `__byte_perm` calls in `i8_table16` with 6x native `__builtin_amdgcn_perm` (`v_perm_b32`) instructions and SGPR immediate constants for `kvalues_iq4nl`.
- **`moe_fused_iq.cu`, `gemm.cu`, `kernels.cu`:** 3-instruction `gfx906` `signed8` decoding and `gfx906` prefill GEMM/kernel optimizations.

### 4. CPU AVX2 Cold-Expert & Thread Pool Optimizations (`engine/src/kernels/cpu/`)
- **`iq_avx2_rows.inl`:** Optimized `iq4nl_rows<NT>` (`IQ4_NL` down projection) by hoisting FP16-to-FP32 activation scales out of the row loop and unifying 128-bit shuffles into a single 256-bit `_mm256_shuffle_epi8` + `_mm256_abs_epi8` (**-11.8%** at 1 token, bitwise identical).
- **`kq_avx2.cpp`:** Rewrote `q4k_dot` into register-pressure-safe templated `q4k_dot_t<NT>` with a `j`-outer loop (eliminating 16-YMM register spills on AVX2) and hoisted activation metadata (`q8s` `_mm_hadd_epi16` / `yd` for `Q4_K`, `dy` / `ym` for `Q5_1`, `dy` + `_mm256_abs_epi8` for `Q8_0`) out of the row loops — **bit-exact** against `ggml-cpu` (`-15.8%` mean / `-18.3%` p90 on `Q4_K / Q5_1`, `-6.8%` mean on `Q4_K / Q8_0`).
- **`native_expert.cpp`:** Enabled 1-token multi-token AVX2 dispatch (`mt_min = 1`) and `STRATA_KQ256` by default on AVX2-only (non-AVX-512) CPUs (`STRATA_KQ256=0` still disables).
- **`pool.cpp`:** Enforced 32/16-row cache-line-aligned task granularity in `ExpertPool` to reduce false sharing and empty tasks across worker threads.

---

## Bitwise Parity Verification (`20/20 Passed`)

Every optimization is gated by upstream's exact parity test suites:

```bash
ctest --test-dir build-gfx906-0.2.20 \
  -R "iq_multi_parity|native_grouped_parity|native_grouped_fused_parity|native_i8_parity|mmvf_rows_parity|gdn_parity|iq_avx2_parity|native_arithmetic_parity" \
  --output-on-failure
```

Verified `100% tests passed, 0 tests failed out of 20`: `iq_multi_parity` (4 suites), `native_grouped_parity`, `native_i8_parity`, `mmvf_rows_parity`, `native_grouped_fused_parity`, `gdn_parity` (2 suites), `iq_avx2_parity` (2 suites), and `native_arithmetic_parity` (all 8 suites including `kq256`, `mt1`, `avx`, `nvfp4`).

---

## Verified A/B Benchmarks (Stock `v0.2.20` vs. Patched `v0.2.20`)

**Test Rig:** HPE CL2200 G3 (FOXCONN Hercules-B) | Dual Intel Xeon E5-2673 v4 (Node 1: 4x16 GB DDR4-2133 Quad-Channel, 100 W BIOS cap, `performance` + `EPB=0`; Node 0: `schedutil`) | AMD Radeon Pro VII (`gfx906`, PCIe 3.0 x16 `MaxPayload 256B` `12.47 GB/s`, 150 W power cap) | Model: `Qwen3.8-Flash-Next IQ3_S` (`--kv int8`, `--kv-resident 6144`, `--pcie-frac 0.2`, `STRATA_ROUTE_TAIL_SKIP=0`).

### GPU & CPU Kernel Microbenchmarks (150 W GPU / Xeon E5-2673 v4 Node 1)

| Kernel / Workload | Stock `v0.2.20` | Patched `v0.2.20` | Speedup |
| --- | --- | --- | --- |
| GPU `IQ3_S iq_mmvq` `8192x2560` (`ncols = 1 / 2 / 4 / 8`) | 50.6 / 65.2 / 91.9 / 144.3 us | **32.9 / 42.9 / 69.2 / 123.4 us** | **1.17x – 1.54x** |
| GPU `IQ3_XXS iq_mmvq` `8192x2560` (`ncols = 1 / 2 / 4 / 8`) | 48.3 / 62.8 / 87.6 / 142.7 us | **24.2 / 40.4 / 64.1 / 119.3 us** | **1.20x – 2.00x** |
| GPU `IQ3_S / Q2_0` grouped `EXP_MODE=8` (`m = 1 / 4 / 8`) | 237.0 / 201.1 / 374.5 us | **147.1 / 195.8 / 369.7 us** | **1.61x at `m=1` (-37.9%)** |
| GPU `IQ3_XXS / Q2_0` grouped `EXP_MODE=8` (`m = 1 / 4 / 8`) | 140.0 / 198.5 / 374.5 us | **112.8 / 182.7 / 343.5 us** | **1.09x – 1.24x** |
| GPU `native_bf16.cu` (`mmvf_multi` / `mmvf_rows`) | 49 VGPR spills / 5,010 `ds_bpermute` | **0 spills / 5,010 `ds_swizzle`** | Clean 0-spill BF16 |
| CPU AVX2 `IQ4_NL down` (1 thread, 1 / 2 tokens) | 0.212 / 0.322 ms | **0.187 / 0.295 ms** | **1.09x – 1.13x** |
| CPU AVX2 `IQ3_S/IQ4_NL engine` (1 tok, default env) | 1.453 ms | **0.835 ms** | **1.74x** |
| CPU AVX2 `Q4_K / Q5_1` pool (`18w+1h`, `nt1=0.78`, FNV match) | 0.6142 mean / 0.8195 p90 ms | **0.5170 mean / 0.6697 p90 ms** | **1.19x mean / 1.22x p90** |
| CPU AVX2 `Q4_K / Q8_0` pool (`18w+1h`, `nt1=0.78`, FNV match) | 0.5895 mean / 0.7572 p90 ms | **0.5496 mean / 0.7008 p90 ms** | **1.07x mean / 1.08x p90** |

### End-to-End `strata_bench` (`IQ3_S`, `--kv int8`, `STRATA_ROUTE_TAIL_SKIP=0`)

| Metric | Stock `v0.2.20` | Patched `v0.2.20` | Change |
| --- | --- | --- | --- |
| **Chat Decode `q1` (exact 38-window match)** | 40.70 tok/s (`64.58 ms/win`) | **41.50 tok/s (`63.39 ms/win`)** | **-1.19 ms/win (+0.80 tok/s)** |
| **Chat Decode** median (range / mean, `n=9`) | 35.90 (31.70–40.70 / 35.76) tok/s | **35.80–36.60 (32.80–41.50 / 37.11) tok/s** | **Up to +3.8% mean, 41.5 peak** |
| **Chat TTFT** median (min) | 0.97 s (0.93 s) | **0.93–0.94 s (0.86–0.87 s)** | **-4.1% (-7.5% min)** |
| **Agent Decode** median (range, `n=15`) | 38.40 (34.70–44.80) tok/s | **38.30–39.10 (33.80–45.70) tok/s** | **Up to 45.70 tok/s peak** |
| **1-Client Total** median (peak, `n=3`) | 23.85 (24.36) tok/s | **24.07–24.58 (24.52–26.20) tok/s** | **Up to +3.1% med / +7.6% peak** |
| **4K / 16K Prompt** peak | 569.20 / 598.50 tok/s | **567.30 / 596.40 tok/s** | Parity (~7.14 s / ~27.0 s) |

---

## Building for `gfx906` (ROCm)

> **Important:** Keep `-DSTRATA_ENABLE_HIP=OFF` (do not combine with `STRATA_HIP_GFX906=ON`) and disable `inkling2` (`-DSTRATA_BUILD_INKLING=OFF -DSTRATA_BUILD_INKLING2=OFF`) so CMake does not search for `nvcc`.

```bash
cmake -S engine -B build-gfx906-0.2.20 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_BUILD_INKLING=OFF \
  -DSTRATA_BUILD_INKLING2=OFF \
  -DSTRATA_BUILD_TESTS=ON

ninja -C build-gfx906-0.2.20 -j8 strata strata-device
```

### Recommended System Tuning (Linux / Dual-Socket NUMA)
- **PCIe MaxPayload 256B:** Add `pci=pcie_bus_perf` to `GRUB_CMDLINE_LINUX_DEFAULT` (`11.3 -> 12.5 GB/s` on PCIe 3.0 x16).
- **NUMA & CPU Governor:** Bind Strata with `numactl --cpunodebind=<gpu_node> --preferred=<gpu_node>` and set `performance` governor + `energy_perf_bias=0` on the GPU's NUMA node only (leaving the idle socket on `schedutil` saves ~19 W).
- **VRAM Headroom with `--kv int8` (16 GB Card):** Pair `--kv int8` with `--kv-resident 6144 --vram-reserve-mib 448` to keep all `102 slots/layer` (`4896` total) while maintaining `> 512 MiB` free VRAM (avoiding ROCm paging warnings).

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | [Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). [Coder: IQ1\_M](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) ([guide](docs/MODELS.md#coder)) for coding. |
| **Unsloth** | [UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) ([guide](docs/MODELS.md#swift-15)), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md#qwen38-flash-next).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** [UD-Q4\_K\_XL, UD-Q6\_K\_XL, Q8\_0](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><b>Tencent Hy4-Flash</b></summary>

| Provider | Choices |
| --- | --- |
| **Strata native (`.strata`)** | [SQ-64, SQ-96, SQ-128](https://huggingface.co/strata-engine/Hy4-Flash-Strata) (with Engram shard on SSD, DFly2 drafter, and vision; base model: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash)). |

[Compare sizes, memory needs and compatibility](docs/MODELS.md#tencent-hy4-flash).

</details>

<details>
<summary><b>Inkling-2-Mini</b></summary>

| Provider | Choices |
| --- | --- |
| **Strata native (`.strata`)** | [SQ-10, SQ-15, SQ-20](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) (with built-in DSpark drafter and vision; base model: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)). |
| **Unsloth GGUF** | [UD-IQ1\_S through UD-Q8\_K\_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (with companion DSpark drafter and vision). |

[Compare sizes, memory needs and compatibility](docs/MODELS.md#inkling-2-mini).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.20-windows-x64-setup.exe` · `Strata-0.2.20-linux-x86_64.AppImage` · `Strata-0.2.20-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (generated from the public GitHub sponsors: one row per level, the biggest supporters first and largest; private sponsors are not listed) -->
<a href="https://github.com/codeconduit-official"><img src="https://images.weserv.nl/?url=github.com/codeconduit-official.png&w=240&h=240&fit=cover&mask=circle&output=png" width="120" height="120" alt="codeconduit-official" title="codeconduit-official"></a>

<a href="https://github.com/kod88vn"><img src="https://images.weserv.nl/?url=github.com/kod88vn.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/router0mail"><img src="https://images.weserv.nl/?url=github.com/router0mail.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="router0mail" title="router0mail"></a>
<a href="https://github.com/kingdonb"><img src="https://images.weserv.nl/?url=github.com/kingdonb.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="kingdonb" title="kingdonb"></a>
<a href="https://github.com/Daisame"><img src="https://images.weserv.nl/?url=github.com/Daisame.png&w=160&h=160&fit=cover&mask=circle&output=png" width="80" height="80" alt="Daisame" title="Daisame"></a>

<a href="https://github.com/ryan4yin"><img src="https://images.weserv.nl/?url=github.com/ryan4yin.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://images.weserv.nl/?url=github.com/themactep.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://images.weserv.nl/?url=github.com/macintog.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://images.weserv.nl/?url=github.com/bisonbet.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://images.weserv.nl/?url=github.com/vizv.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="vizv" title="vizv"></a>
<a href="https://github.com/seoulworker"><img src="https://images.weserv.nl/?url=github.com/seoulworker.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jaceknoga"><img src="https://images.weserv.nl/?url=github.com/jaceknoga.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jaceknoga" title="jaceknoga"></a>
<a href="https://github.com/jackyini"><img src="https://images.weserv.nl/?url=github.com/jackyini.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/moruku36"><img src="https://images.weserv.nl/?url=github.com/moruku36.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="moruku36" title="moruku36"></a>
<a href="https://github.com/hrkgk"><img src="https://images.weserv.nl/?url=github.com/hrkgk.png&w=112&h=112&fit=cover&mask=circle&output=png" width="56" height="56" alt="hrkgk" title="hrkgk"></a>
<!-- supporters:end -->

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

