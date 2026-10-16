# Sammie-Roto 2 — panda fork (2.3.3+panda.2)

A personal fork of [Sammie-Roto 2](https://github.com/Zarxrax/Sammie-Roto-2) by Zarxrax, based on version **2.3.3**. It adds segmentation and matting models to the original app. Everything else works as upstream: see the original README below.

> **Models are not included and mostly not downloaded for you.** Several of them are gated on Hugging Face, so you need to accept their licenses and download them yourself. The [Models](#models) section lists every file and where it goes.

Tested on Rocky Linux with NVIDIA GPUs (RTX 4060 8 GB and RTX 5000 32 GB, CUDA 13.0, PyTorch 2.10). The additions have not been tested on Windows or macOS.

## What's new

**2.3.3+panda.2**
- **LTX Alpha matting.** A new matting model built on Lightricks' [LTX-2.5 IC-LoRA Alpha-Gen](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen). Sammie grows and feathers the SAM mask, premultiplies the frame onto black, and the model mattes what remains. The Matting tab has resolution (576 / 736 / 1088 / Full), batch size (49 / 97 / 121 / 145 frames) and crossfade overlap (2 / 4 / 8 / 16 frames) options, like VideoMaMa.
- Runs fully offline: the empty text prompt is pre-encoded once, so the 24 GiB Gemma text encoder is never loaded at runtime.
- Optional fp8 build of the transformer with the LoRA fused in (21.9 GiB instead of 39.1 + 1.2 GiB; a two-batch test ran 2.7× faster).
- Fixed: a fresh install crashed on startup with `Unrecognized SAM2 model name: 'Base'`.

**2.3.3+panda.1**
- **SAM3** segmentation model.
- **ViTMatte** (S and B, DIS) and **MEMatte** (ViT-S, ViT-B, and the real-world model) image matting models.

## Installation (Linux)

```bash
git clone https://github.com/lukaszomasta/Sammie-Roto-Panda.git
cd Sammie-Roto-Panda
bash install_dependencies.sh
./run_sammie.command
```

The installer creates `venv/`, installs PyTorch (choose CUDA 13.0 for RTX cards), the requirements and detectron2 (needs a C++ compiler), and offers to install the LTX-2 code (see [LTX Alpha setup](#ltx-alpha-setup)).

## Models

All paths are relative to the Sammie folder.

### Upstream models

SAM2, EfficientTAM, MatAnyone, MatAnyone2, MiniMax-Remover and VideoMaMa work as in upstream: Sammie offers to download them the first time you use them.

### SAM3 (panda.1)

Gated: request access at [facebook/sam3](https://huggingface.co/facebook/sam3) first.

| File | Put it in |
|---|---|
| `config.json`, `model.safetensors`, `merges.txt`, `processor_config.json` | `checkpoints/sam3/` |

### ViTMatte and MEMatte (panda.1)

| Model | Files | Source | Put it in |
|---|---|---|---|
| ViTMatte | `ViTMatte_S_DIS.pth`, `ViTMatte_B_DIS.pth` | [nielsr/vitmatte-checkpoints](https://huggingface.co/nielsr/vitmatte-checkpoints) | `checkpoints/vitmatte/` |
| MEMatte | `MEMatte_ViTS_DIM.pth` ([Google Drive](https://drive.google.com/file/d/122p3sdhJVb7vg4IXELeC9C3HEG9Mlh5z/view)), `MEMatte_ViTB_DIM.pth` ([Google Drive](https://drive.google.com/file/d/1NOV64zMSFtoKPASqvEvxQKI_PRY9m5IA/view)), `MEMatte.pth`: the real-world (AIM-500) ViT-S model, saved under this name ([Google Drive](https://drive.google.com/file/d/1R5NbgIpOudKjvLz1V9M9SxXr1ovAmu3u/view)) | [MEMatte repository](https://github.com/linyiheng123/MEMatte) | `checkpoints/mematte/` |

### LTX Alpha (panda.2)

Gated: accept the license on both Lightricks repositories first.

| File | Source | Size (GiB) | Needed for |
|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 39.1 | setup (or runtime without the fp8 build) |
| `ltx-2.5-video-vae-bf16.safetensors` | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 1.4 | runtime |
| `ltx-2.5-audio-vae-bf16.safetensors` | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 0.3 | runtime (loaded, never used) |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 24.5 | setup only |
| `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` | [Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen) | 1.2 | setup (or runtime without the fp8 build) |

Put everything except Gemma in `checkpoints/ltx_alpha/`. Gemma can live anywhere; it's only read once.

## LTX Alpha setup

**Shortcut: prebuilt files.** [lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused](https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused) (gated: accept the LTX-2.x license there) has the fp8 transformer with the LoRA already fused and the encoded empty prompt. Put `ltx-2.5-22b-distilled-alpha-gen-0.9-fp8-fused.safetensors` and `empty_prompt_embeddings.safetensors` in `checkpoints/ltx_alpha/` together with the two VAEs from Lightricks/LTX-2.5, do step 1 below, and skip steps 2 and 3. You then don't need the bf16 transformer, the LoRA or Gemma.

To build the files yourself instead:

1. **Install the LTX-2 code.** Answer `y` when `install_dependencies.sh` asks, or run:
   ```bash
   LTX_REF=be6d6d0a69d11a0a399c462ed92606692e5f8697
   venv/bin/pip install --no-deps \
     "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-core" \
     "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-pipelines"
   ```
   This is the exact LTX-2 revision Sammie was tested with (v1.4.2 + 2 commits).

2. **Encode the empty prompt** (once; CPU, about 23 GB RAM and 6–8 minutes):
   ```bash
   venv/bin/python tools/encode_ltx_empty_prompt.py --gemma /path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
   ```
   This writes `checkpoints/ltx_alpha/empty_prompt_embeddings.safetensors`. After this, you can delete the Gemma file.

3. **Optional, recommended: build the fp8 transformer** with the LoRA fused in (about 2.5 minutes with a CUDA GPU; needs about 32 GB RAM):
   ```bash
   venv/bin/python tools/build_ltx_fp8_fused.py
   ```
   This writes `checkpoints/ltx_alpha/ltx-2.5-22b-distilled-alpha-gen-0.9-fp8-fused.safetensors`. Sammie uses it automatically when present, and you can then delete the bf16 transformer and the LoRA. The mattes match the unfused model within about 0.6/255 on average.

4. Put a copy of the [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) next to the weights.

When everything is in place, `checkpoints/ltx_alpha/` holds the video VAE, the audio VAE, `empty_prompt_embeddings.safetensors`, and either the fp8 fused transformer or the bf16 transformer + LoRA. If something is missing, Sammie lists exactly what when you load the model.

**Hardware.** GPUs with 30 GiB of VRAM or more keep the whole model on the GPU. Smaller GPUs stream the transformer from system RAM. An 8 GB RTX 4060 with 62 GB RAM runs a 992×576 crop in batches of 49 frames (about 2.5 minutes per batch, 5.1 GiB VRAM peak). Resolutions of 1088 and above may run out of memory.

## Licenses

- **Sammie-Roto** and this fork's code: [GPL-3.0](LICENSE).
- **LTX-2 code and weights** (Lightricks): [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x). Not included in this repository. Among other terms, organizations with annual revenue of $10M or more need a paid commercial license from Lightricks.
- **ViTMatte** code in `sammie/vitmatte/`: MIT ([LICENSE](sammie/vitmatte/LICENSE)). **MEMatte** code in `sammie/mematte/`: MIT, as stated in its README ([LICENSE](sammie/mematte/LICENSE)).
- **Gemma 4** is released under Apache 2.0. It is only used once, to make the LTX empty-prompt embedding, and is never shipped or loaded by Sammie.
- **SAM3** (Meta) and the upstream models' weights are covered by their own licenses; check each source before use.

This fork is not affiliated with Zarxrax, Lightricks, Meta or Google.

---

*The original Sammie-Roto 2.3.3 README follows.*

# Sammie-Roto 2
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto 2 screenshot](https://github.com/user-attachments/assets/bc2c99c8-4039-49f1-94ed-65f104a83e8d)

[![GitHub Downloads](https://img.shields.io/github/downloads/Zarxrax/Sammie-Roto-2/total)](https://github.com/Zarxrax/Sammie-Roto-2/releases)
[![GitHub Code License](https://img.shields.io/github/license/Zarxrax/Sammie-Roto-2)](LICENSE)
[![Discord](https://img.shields.io/discord/1437589475369811970?label=Discord&color=blue)](https://discord.gg/jb5qrFyGFF)

**Please add a Github Star if you find it useful!**

Sammie-Roto 2 is an easy-to-use, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone), [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), and [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)

Sammie-Roto 2 is free and open source, but runs models produced by several external projects and organizations. Some models may have restrictions on commercial usage. Please check with the relevant model provider if you have questions regarding licensing.

### Updates
**Full Changelog can be seen under [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)**
- [04/17/2026] 2.3.3 - Several large performance optimizations, and colorspace conversions are now handled correctly.
- [04/10/2026] 2.3.2 - Improved temporal stability for VideoMaMa.
- [04/02/2026] 2.3.1 - Added a live preview during segmentation while holding the shift key.
- [03/27/2026] 2.3.0 - Added VideoMaMa model, added option to combine objects when matting, fixed major segmentation bug, and more.
- [03/08/2026] 2.2.0 - Added MatAnyone2 model.
- [01/18/2026] 2.1.1 - Rebuilt the export dialog, slightly faster application startup, bug fixes.
- [12/16/2025] 2.1.0 - Added In/Out markers. Modifying points no longer deletes tracking data. Enabled half-precision for much faster segmentation. Added EfficientTAM model.
- [11/23/2025] 2.0.0 - First stable release. Includes several new features and bugfixes. New quick-start video tutorial and [Discord server](https://discord.gg/jb5qrFyGFF).
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Documentation and Tutorials:
## [Wiki Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

[![Quick Start Video](https://img.youtube.com/vi/m0iZpxsZJcE/0.jpg)](https://www.youtube.com/watch?v=m0iZpxsZJcE)

*This video does not cover new features and changes added since version 2.0

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive to any location that doesn't restrict write permissions (so not in Program Files)
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software.

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder. You can also move the folder.

### Installation (Linux, Mac)
- MacOS users: Make sure Homebrew is installed.
- Ensure [Python](https://www.python.org/) is installed (version 3.10 or higher, 3.12 recommended)
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command: `bash install_dependencies.sh` then follow the prompt.
- MacOS users: double-click "run_sammie.command" to launch the program. Linux users: `bash run_sammie.command` or execute the file however you prefer.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone) & [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone code)
* [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
* Some icons by [Yusuke Kamiyamane](http://p.yusukekamiyamane.com/)
