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

PBR texture generation needs two native modules built **inside the extension
venv** from the vendored repo source (`<MODEL_DIR>/_hy3d21/hy3dpaint/...`):

- `custom_rasterizer` → `custom_rasterizer_kernel` (CUDA/C++) — needs `nvcc` + MSVC
- `DifferentiableRenderer/mesh_inpaint_processor.cpp` → `mesh_inpaint_processor` (C++) — needs MSVC

The generator raises a clear error with build instructions if `custom_rasterizer_kernel`
is missing. **Shape generation works without them.**

PBR also needs these pip packages (installed by `setup.py`): `xatlas`,
`pygltflib`, `bpy==4.2.0` (Blender as a module — locked to the interpreter's
Python minor version), `basicsr`, `realesrgan`, `pybind11`.

### Windows build (validated)

> **Automated:** on the first PBR generation the extension applies the source
> patches and compiles both native extensions automatically
> (`generator._build_texgen_extensions`): it locates `vcvars64.bat` (via
> `vswhere`) and the CUDA Toolkit, sets `DISTUTILS_USE_SDK=1`, and builds. If the
> prerequisites below are missing it raises an error with the `winget` commands.
> The steps below document what that automation does (and how to do it by hand).

On Windows the upstream Linux build (`pip install -e .`, `compile_mesh_painter.sh`)
does **not** work as-is. This is the procedure that was validated on
**Python 3.11.9 + torch 2.7.0+cu128**:

**Prerequisites:** Visual Studio Build Tools 2022 (Desktop development with C++ →
`cl.exe`) and CUDA Toolkit 12.8 (`nvcc`, matching torch's `cu128`). Install them
with `winget` from an **elevated** terminal (multi-GB downloads):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Nvidia.CUDA -e --version 12.8
```

After installing, **open a new terminal** so `cl.exe` and `nvcc` are picked up
(or set `CUDA_PATH` and load `vcvars64.bat` manually, as below).

**Build environment** (in one shell, using the venv's `python`/`pip`):

1. Load MSVC x64 env from `…\BuildTools\VC\Auxiliary\Build\vcvars64.bat`.
2. Set `CUDA_PATH`/`CUDA_HOME` to `…\CUDA\v12.8` and prepend its `bin` to `PATH`.
3. Set `DISTUTILS_USE_SDK=1` (torch requires this when the VC env is pre-activated).
4. Ensure `wheel` and `pybind11` are installed in the venv.

**custom_rasterizer** (from `_hy3d21/hy3dpaint/custom_rasterizer`):

```
python setup.py build_ext --inplace
pip install . --no-build-isolation --no-deps
```

> Do **not** use `pip install -e .`: setuptools ≥ 80's deprecated `develop`
> command re-invokes the build with isolation and loses `torch`
> (`ModuleNotFoundError: No module named 'torch'`).

**mesh_inpaint_processor** (from `_hy3d21/hy3dpaint/DifferentiableRenderer`):
only a Linux `compile_mesh_painter.sh` ships. Build the `.pyd` in-place with a
tiny pybind11 setup script (`Pybind11Extension("mesh_inpaint_processor",
["mesh_inpaint_processor.cpp"], cxx_std=17)` + `build_ext --inplace`). The
module is imported relatively (`from .mesh_inpaint_processor import …`), so the
`.pyd` must stay in the `DifferentiableRenderer` folder.

**Two MSVC source patches are required** in
`custom_rasterizer/lib/custom_rasterizer_kernel/` (g++/Linux accepts these, MSVC
does not):

- *C2398 (narrowing):* wrap the `size()` arguments to `torch::zeros({…})` with
  `(int64_t)` in `grid_neighbor.cpp`.
- *LNK2001 `data_ptr<long>`:* on Windows `long` is 32-bit and torch only
  instantiates `data_ptr<int64_t>`. Replace every `data_ptr<long>` with
  `data_ptr<int64_t>` (and the receiving `long*` with `int64_t*`) in
  `grid_neighbor.cpp`, `rasterizer.cpp`, `rasterizer_gpu.cu`.

> ⚠️ These patches live in the **vendored** source under the model dir, not in
> this repo. If the model dir / `_hy3d21` is wiped (re-download), re-apply them
> before rebuilding. Once built, the compiled `.pyd`s are installed into the
> venv and survive. The community fork `lzz19980125/Hunyuan3D-2.1-Windows`
> carries equivalent fixes.

## Troubleshooting

- If installation fails after changing installer logic, run **Repair** in Modly
  so the extension venv is recreated.
- If texture generation fails with an import/OSError about
  `textureGenPipeline` / `custom_rasterizer`, build the native extensions as
  described above.
- Out-of-memory during texture generation → lower **Texture Views** (6) and
  **Texture Resolution** (512), or disable PBR texture.
- `DLL load failed while importing bpy` during texture generation: `bpy`'s
  bundled DLLs conflict with an already-loaded torch/CUDA in the same process.
  This extension handles it by importing `bpy` defensively in the vendored
  `DifferentiableRenderer/mesh_utils.py` (try/except) and running the OBJ→GLB
  conversion in a **separate subprocess** (`generator._obj_to_glb`), where bpy
  loads cleanly. If the vendored source is re-downloaded, the `mesh_utils.py`
  try/except patch must be re-applied.

## Upstream model sources

- Model weights: `tencent/Hunyuan3D-2.1`
- Project source: `Tencent-Hunyuan/Hunyuan3D-2.1`
