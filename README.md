# modly-hunyuan3d-2-1-extension

Modly extension for **Hunyuan3D 2.1**, Tencent's full image-to-3D pipeline with
production-ready **PBR** material generation.

This is a separate extension from `modly-hunyuan3d-mini-extension`. The Mini
extension (Hunyuan3D 2.0 Mini, ~6 GB VRAM) is left untouched; install whichever
one matches your hardware.

## What this extension does

- installs an isolated Python environment for the extension
- loads the Hunyuan3D **2.1** shape pipeline (`hy3dshape`) inside Modly
- optionally generates **PBR textures** with the Hunyuan3D-Paint 2.1 pipeline
  (`hy3dpaint`) when the required native extensions are available

## ⚠️ Hardware requirements

Hunyuan3D 2.1 has **no Mini variant** — it is the full model:

| Stage                 | Approx. VRAM |
|-----------------------|--------------|
| Shape generation      | ~10 GB       |
| PBR texture (paint)   | ~21 GB       |
| Shape + texture       | ~29 GB       |

If you only have ~6–8 GB of VRAM, use the Mini extension instead.

## Model / source layout

| Item                     | Location                                                |
|--------------------------|---------------------------------------------------------|
| Shape weights            | `tencent/Hunyuan3D-2.1` → `hunyuan3d-dit-v2-1`          |
| PBR paint weights        | `tencent/Hunyuan3D-2.1` → `hunyuan3d-paintpbr-v2-1`     |
| Python source (vendored) | `_hy3d21/hy3dshape` + `_hy3d21/hy3dpaint` (from GitHub) |
| Upscaler                 | `RealESRGAN_x4plus.pth` (downloaded on first texture)  |

Weights and source are downloaded lazily on first use, so the first generation
(and first textured generation) takes longer.

## Installation flow

At install time, `setup.py` creates a virtual environment and selects the
PyTorch stack from the platform information Modly passes in (`gpu_sm`,
`cuda_version`, OS / CPU architecture). Hunyuan3D 2.1 is validated on
PyTorch 2.5.1; newer CUDA builds are used only where the GPU requires them.

## PBR texture: native extensions

PBR texture generation needs two native components built **inside the extension
venv** from the vendored repo source:

- `_hy3d21/hy3dpaint/custom_rasterizer`  →  `pip install -e .`
- `_hy3d21/hy3dpaint/DifferentiableRenderer`  →  `bash compile_mesh_painter.sh`

The generator raises a clear error with the exact build commands if they are
missing. **Shape generation works without them.**

### Windows note

The 2.1 texture extensions require MSVC build tools and have known Windows
build issues. If `custom_rasterizer` / `DifferentiableRenderer` fail to compile
on Windows, consult the community Windows fork
(`lzz19980125/Hunyuan3D-2.1-Windows`). Shape-only generation is unaffected.

## Troubleshooting

- If installation fails after changing installer logic, run **Repair** in Modly
  so the extension venv is recreated.
- If texture generation fails with an import/OSError about
  `textureGenPipeline` / `custom_rasterizer`, build the native extensions as
  described above.
- Out-of-memory during texture generation → lower **Texture Views** (6) and
  **Texture Resolution** (512), or disable PBR texture.

## Upstream model sources

- Model weights: `tencent/Hunyuan3D-2.1`
- Project source: `Tencent-Hunyuan/Hunyuan3D-2.1`
