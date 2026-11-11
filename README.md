# Strata (Custom Hardware-Optimized Fork)

A personal fork of [Strata](https://github.com/Niko1221/Strata/) optimized for budget, power-constrained heterogeneous hardware (asymmetrical dual-socket NUMA and AMD ROCm Instinct architecture). Tuned and optimized with assistance from Gemini Argon.

---

## 🖥 Target Hardware Profile

This build is specifically tuned for a power-capped, asymmetrical multi-socket homelab/workstation configuration:

| Component | Specification | Power Limit / Tuning Notes |
| :--- | :--- | :--- |
| **CPUs** | Dual Intel Xeon E5-2673 v4 (Broadwell-EP) | Capped at **65W TDP** per socket |
| **GPU** | 1× AMD Radeon Instinct MI50 (16GB HBM2, PCIe 3.0 via CPU0) | Capped at **130W** via ROCm / powerPlay |
| **CPU0 Memory** | 3× 16GB DDR4-2133 ECC REG (48 GB) | Primary host memory pool & GPU DMA host |
| **CPU1 Memory** | 1× 16GB DDR4-2133 ECC REG (16 GB) | Secondary NUMA node / offload tasks |
| **Total Memory** | 64 GB DDR4-2133 (Asymmetric 48GB / 16GB NUMA split) | Local node allocation priority on Node 0 |

---

## ⚡ Key Optimizations

- **NUMA & Socket Affinity:** Thread affinity and host memory allocations are pinned to **Node 0** to eliminate inter-socket UPI/QPI traffic and PCIe bottlenecks with the MI50.
- **Power-Cap Resilience:** Thread pooling, execution scheduling, and batch sizes are calibrated to prevent frequency drops under the strict 65W CPU and 130W GPU limits.
- **ROCm / HBM2 KV Offloading:** Layer distribution and KV cache reuse are balanced to fit within the MI50's 16GB HBM2 while falling back gracefully to Node 0 DDR4 memory when needed.

---

## 📊 Benchmark Results

- **Model Tested:** `Q3.8 FN`
- **Quantization:** `IQ3_S`
- **Metric Baseline:** Reported as median values alongside min–max ranges over multiple test runs.

### 1. Interactive Chat & Latency

| Metric | Median | Range (Min – Max) | Samples ($n$) |
| :--- | :--- | :--- | :--- |
| **Chat Decode Speed** | **27.80 tok/s** | 24.60 – 30.60 tok/s | $n = 9$ |
| **Chat Time to First Token (TTFT)** | **1.60 s** | 1.48 – 1.72 s | $n = 9$ |

### 2. Long-Context Prompt Processing

| Context Length | Prompt Processing Speed | Time to First Token (TTFT) | Samples ($n$) |
| :--- | :--- | :--- | :--- |
| **4K Context** | **477.00 tok/s** *(409.10 – 478.60)* | **8.49 s** *(8.46 – 9.90)* | $n = 3$ |
| **16K Context** | **527.70 tok/s** *(505.50 – 527.90)* | **30.51 s** *(30.50 – 31.85)* | $n = 3$ |

### 3. Agentic & Tool-Calling Workloads

| Metric | Median | Range (Min – Max) | Samples ($n$) |
| :--- | :--- | :--- | :--- |
| **Agent Cache Hit Rate** | **69.30%** | 69.09% – 69.48% | $n = 3$ |
| **Agent Re-read Tokens / Session** | **73.00 tokens** | 4.00 – 143.00 tokens | $n = 3$ |
| **Agent Turn TTFT** | **7.17 s** | 4.61 – 10.44 s | $n = 15$ |
| **Agent Decode Speed** | **31.00 tok/s** | 27.50 – 34.40 tok/s | $n = 15$ |

### 4. End-to-End Throughput

| Concurrency | Total Throughput | Range (Min – Max) | Samples ($n$) |
| :--- | :--- | :--- | :--- |
| **1 Client Total** | **18.76 tok/s** | 15.29 – 18.96 tok/s | $n = 3$ |

---

## 🛠 Recommended Launch & Environment Configuration

To replicate the affinity and power settings used in these runs:

```bash
# 1. Set GPU Power Cap to 130W (Instinct MI50)
sudo rocm-smi --setpoweroverdrive 130

# 2. Bind execution strictly to NUMA Node 0 (Host of the MI50 GPU)
numactl --cpunodebind=0 --membind=0 ./strata-server ...

# 3. config.json
{
  "exe": "/path/to/strata",
  "cwd": "./",
  "log": "strata.log",
  "host": "0.0.0.0",
  "fit_max_tokens": true,
  "tokenizer": "/pack/iq3s-ablit/tokenizer",
  "args": [
    "--pack", "/pack/iq3s-ablit",
    "--native", "/path/to/00001-of-00002.gguf",
    "--ple-gguf", "/path/to/00002-of-00002.gguf",
    "--conversation-cache-mib", "1024",
    "--max-context", "131072",
    "--expert-profile", "data/expert-profile.bin",
    "--kv", "k8v4",
    "--kv-resident", "16384",
    "--spec", "3",
    "--mtp", "/mtp/rt",
    "--spec-min-p", "0.5",
    "--pcie-frac", "0.2",
    "--vram-reserve-mib", "544",
    "--prefill", "auto",
    "--expert-cache", "auto",
    "--expert-cache-per-layer",
    "--adapt-every", "1",
    "--adapt-swaps", "80",
    "--adapt-decay", "0.97",
    "--adapt-async", "1"
  ],
  "backend": "hip",
  "env": {
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_IQ_MT_MIN": "1",
    "STRATA_CPU_MT1": "1",
    "STRATA_IQ256_GATHER": "1",
    "STRATA_HIP_ADAPT_KERNEL_COPY": "1",
    "STRATA_ADAPT_COPY_BLOCKS": "16",
    "STRATA_DENSE_MMQ": "1",
    "STRATA_HIP_PROMPT_F16": "1",
    "STRATA_LFUSE": "1",
    "STRATA_DECODE_TIMING": "1",
    "STRATA_EXP_MODE": "8",
    "STRATA_MMVF_ROWS": "1",
    "STRATA_GR_NORM_FAST": "1",
    "STRATA_VERIFY_QDEDUP": "1",
    "STRATA_POST_COPY_WIDE": "1",
    "STRATA_GR_UP_EXACT_T": "1",
    "STRATA_SELECT_SIMT": "1",
    "STRATA_SESSION_DELTA": "1",
    "STRATA_PREFILL_CPU_SHARE": "auto",
    "STRATA_PREFILL_CPU_TOKENS": "16",
    "STRATA_PREFILL_CPU_SHARE_MAX": "8192"
  },
  "api_monitor": true,
  "gpu_temp_sensor": "junction",
  "qwen_code_tail": true
}

```

---

## 📌 Acknowledgments

- Upstream project: **Strata**
- Optimization and profiling guidance: **Gemini Argon**


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
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
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (IST-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more, text only ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.9-windows-x64-setup.exe` · `Strata-0.2.9-linux-x86_64.AppImage` · `Strata-0.2.9-macos-arm64.dmg`</sub>

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
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
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
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A coffee helps fund development and testing on more hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

