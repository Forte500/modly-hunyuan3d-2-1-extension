"""
Hunyuan3D 2.1 — Modly generator.

Reference : https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
            https://huggingface.co/tencent/Hunyuan3D-2.1

Differences from the 2.0 Mini extension this is based on:
  * No "mini" variant exists for 2.1 — this loads the full DiT v2.1 model
    (~10 GB VRAM for shape, ~21 GB for PBR texture).
  * The Python package was split/renamed: shape lives in `hy3dshape`
    (was `hy3dgen.shapegen`) and texture in `hy3dpaint` (was `hy3dgen.texgen`).
  * Texture generation produces PBR materials and operates on *file paths*
    (mesh in -> textured mesh out), not in-memory meshes.
  * The paint pipeline is configured through `Hunyuan3DPaintConfig` and needs
    the RealESRGAN_x4plus.pth upscaler weights.
"""
import io
import os
import random
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID      = "tencent/Hunyuan3D-2.1"
_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"
_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"
_GITHUB_ZIP      = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/archive/refs/heads/main.zip"
_REALESRGAN_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Folders vendored from the 2.1 GitHub repo into the model dir.
_VENDOR_DIR_NAME = "_hy3d21"

# Source patches required to build the native PBR texture extensions and to keep
# the paint pipeline importable. The C++ ones fix MSVC strictness that g++/Linux
# tolerates; the mesh_utils one makes `bpy` optional in-process. Applied
# idempotently (paths are relative to the vendored `hy3dpaint` dir).
_CR_KERNEL = "custom_rasterizer/lib/custom_rasterizer_kernel/"
_NATIVE_PATCHES = {
    # bpy clashes with torch/CUDA DLLs once loaded in-process → make it optional
    # (obj->glb runs in a clean subprocess instead).
    "DifferentiableRenderer/mesh_utils.py": [
        ("import os\nimport cv2\nimport bpy\nimport math\nimport numpy as np",
         "import os\nimport cv2\ntry:\n    import bpy\nexcept Exception:\n    bpy = None\nimport math\nimport numpy as np"),
    ],
    # C2398: narrowing size_t -> int64 inside torch::zeros({...}) brace-init.
    # LNK2001: Windows `long` is 32-bit; torch only instantiates data_ptr<int64_t>.
    _CR_KERNEL + "grid_neighbor.cpp": [
        ("torch::zeros({seq2pos.size() / 3, 3}", "torch::zeros({(int64_t)(seq2pos.size() / 3), 3}"),
        ("torch::zeros({seq2pos.size() / 3}", "torch::zeros({(int64_t)(seq2pos.size() / 3)}"),
        ("torch::zeros({seq2feat.size() / feat_channel, feat_channel}", "torch::zeros({(int64_t)(seq2feat.size() / feat_channel), feat_channel}"),
        ("torch::zeros({grids[i].seq2grid.size(), 9}", "torch::zeros({(int64_t)grids[i].seq2grid.size(), 9}"),
        ("torch::zeros({grids[i].seq2evencorner.size()}", "torch::zeros({(int64_t)grids[i].seq2evencorner.size()}"),
        ("torch::zeros({grids[i].seq2oddcorner.size()}", "torch::zeros({(int64_t)grids[i].seq2oddcorner.size()}"),
        ("torch::zeros({grids[i].downsample_seq.size()}", "torch::zeros({(int64_t)grids[i].downsample_seq.size()}"),
        ("long* nptr =", "int64_t* nptr ="),
        ("long* dptr =", "int64_t* dptr ="),
        (".data_ptr<long>()", ".data_ptr<int64_t>()"),
    ],
    _CR_KERNEL + "rasterizer.cpp": [
        (".data_ptr<long>()", ".data_ptr<int64_t>()"),
    ],
    _CR_KERNEL + "rasterizer_gpu.cu": [
        (".data_ptr<long>()", ".data_ptr<int64_t>()"),
    ],
}


class Hunyuan3D21Generator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-2-1"
    DISPLAY_NAME = "Hunyuan3D 2.1"
    VRAM_GB      = 10  # shape only; PBR texture needs ~21 GB

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SHAPE_SUBFOLDER
        model_dir = self.model_dir / subfolder
        return model_dir.exists() and (
            (model_dir / "model.fp16.safetensors").exists()
            or (model_dir / "model.safetensors").exists()
            or (model_dir / "model.fp16.ckpt").exists()
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3d21()

        import torch
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float32  # MPS has limited fp16 op coverage
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

        subfolder = self.download_check if self.download_check else _SHAPE_SUBFOLDER
        print(f"[Hunyuan3D21Generator] Loading shape pipeline from {self.model_dir} (subfolder={subfolder})…")
        # Hunyuan3D-2.1 ships the DiT shape model as `model.fp16.ckpt` only
        # (no safetensors). use_safetensors=True would make from_single_file look
        # for a non-existent `.safetensors` file, so load the ckpt directly.
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=False,
            variant="fp16",
            device=device,
            dtype=dtype,
        )
        self._model = pipeline
        print(f"[Hunyuan3D21Generator] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        # `enable_texture` comes from a True/False select, which may arrive as a
        # bool, an int, or a string ("true"/"false") depending on the UI — parse
        # robustly (note: bool("false") is True, so a plain bool() won't do).
        _tex_raw = params.get("enable_texture", False)
        enable_texture = _tex_raw in (True, 1, "1", "true", "True", "yes", "on")
        octree_res     = int(params.get("octree_resolution", 380))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        shape_end = 65 if enable_texture else 82
        self._report(progress_cb, 12, "Generating 3D shape…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if enable_texture:
            self._report(progress_cb, 67, "Freeing VRAM for PBR texture model…")
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, params, progress_cb)
            self.load()  # restore shape model so next generation doesn't crash
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 85, "Optimizing mesh…")
                mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            # cuDNN/CUDA incompatibility — fall back to CPU
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _run_texture(self, mesh, image: "Image.Image", params: dict, progress_cb=None):
        """Generate PBR materials for `mesh` using the Hunyuan3D-Paint 2.1 pipeline.

        The 2.1 paint pipeline operates on files: it takes an input mesh path and
        an input image path, and writes a textured mesh. We therefore export the
        shape mesh to a temp GLB, run paint, and load the result back as trimesh.
        """
        import torch
        import trimesh

        self._check_texgen_extensions()

        num_view   = int(params.get("texture_num_view", 6))
        resolution = int(params.get("texture_resolution", 512))

        self._report(progress_cb, 70, "Preparing PBR texture model…")
        self._ensure_paint_weights()
        self._ensure_realesrgan()

        self._report(progress_cb, 74, "Loading PBR texture model…")
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        paint_root = self.model_dir / _VENDOR_DIR_NAME / "hy3dpaint"

        conf = Hunyuan3DPaintConfig(num_view, resolution)
        conf.realesrgan_ckpt_path = str(paint_root / "ckpt" / "RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path   = str(paint_root / "cfgs" / "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline      = str(paint_root / "hunyuanpaintpbr")
        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        work = Path(tempfile.mkdtemp(prefix="hy3d21_paint_"))
        in_mesh  = work / "shape.glb"
        in_image = work / "cond.png"
        out_obj  = work / "textured.obj"
        out_glb  = work / "textured.glb"
        try:
            mesh.export(str(in_mesh))
            image.save(str(in_image))

            self._report(progress_cb, 80, "Generating PBR textures…")
            with torch.no_grad():
                # save_glb=False: the in-process obj->glb path imports bpy, whose
                # bundled DLLs clash with torch/CUDA in this process. We export an
                # OBJ (+ PBR textures) and convert to GLB in a clean subprocess.
                result_path = paint_pipeline(
                    mesh_path=str(in_mesh),
                    image_path=str(in_image),
                    output_mesh_path=str(out_obj),
                    save_glb=False,
                )
            obj_path = Path(result_path) if result_path else out_obj

            self._report(progress_cb, 92, "Packing PBR materials (GLB)…")
            textured = self._obj_to_glb(obj_path, out_glb)
        finally:
            del paint_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            try:
                in_image.unlink()
            except OSError:
                pass

        return textured

    def _obj_to_glb(self, obj_path: Path, glb_path: Path):
        """Convert the textured OBJ to a PBR GLB and return it as a trimesh.

        Uses Blender (bpy) via Hunyuan's `convert_obj_to_glb` to preserve the full
        PBR material (base color + metallic-roughness). bpy is run in a fresh
        subprocess because its native DLLs conflict with an already-loaded
        torch/CUDA in the main process. Falls back to a plain trimesh load (base
        color only) if the subprocess conversion is unavailable.
        """
        import subprocess
        import trimesh

        paint_root = self.model_dir / _VENDOR_DIR_NAME / "hy3dpaint"
        code = (
            "import sys;"
            "sys.path.insert(0, sys.argv[1]);"
            "from DifferentiableRenderer.mesh_utils import convert_obj_to_glb;"
            "convert_obj_to_glb(sys.argv[2], sys.argv[3])"
        )
        try:
            subprocess.run(
                [sys.executable, "-c", code, str(paint_root), str(obj_path), str(glb_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            print(f"[Hunyuan3D21Generator] bpy GLB conversion failed, "
                  f"falling back to trimesh (PBR metallic-roughness may be lost):\n{detail}")

        if glb_path.exists():
            return trimesh.load(str(glb_path), force="mesh")
        return trimesh.load(str(obj_path), force="mesh")

    def _check_texgen_extensions(self) -> None:
        """Ensure the native PBR extensions are compiled — patching the vendored
        source and building them automatically on first use if missing."""
        vendor = self.model_dir / _VENDOR_DIR_NAME / "hy3dpaint"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

        if self._texgen_built(vendor):
            return

        # Build before importing textureGenPipeline/MeshRender, so the relative
        # `from .mesh_inpaint_processor import meshVerticeInpaint` succeeds.
        self._build_texgen_extensions(vendor)

        import importlib
        importlib.invalidate_caches()
        if not self._texgen_built(vendor):
            raise RuntimeError(
                "Native PBR texture extensions are still not importable after an "
                "automatic build attempt. See the build log above for the failing step."
            )

    def _texgen_built(self, vendor: Path) -> bool:
        """True if both native extensions are present (without importing the
        heavier MeshRender, which binds mesh_inpaint at its own import time)."""
        try:
            import custom_rasterizer_kernel  # noqa: F401
        except Exception:
            return False
        dr = vendor / "DifferentiableRenderer"
        built = list(dr.glob("mesh_inpaint_processor*.pyd")) + list(dr.glob("mesh_inpaint_processor*.so"))
        return bool(built)

    def _build_texgen_extensions(self, vendor: Path) -> None:
        print("[Hunyuan3D21Generator] Native PBR extensions missing — patching & building (one-time)…")
        self._patch_native_sources(vendor)
        env = self._build_env()
        pip = [sys.executable, "-m", "pip"]
        self._run_build(pip + ["install", "-q", "wheel", "pybind11", "ninja"],
                        vendor, env, "install build tooling")

        cr = vendor / "custom_rasterizer"
        self._run_build([sys.executable, "setup.py", "build_ext", "--inplace"],
                        cr, env, "compile custom_rasterizer")
        # Non-editable install: setuptools' deprecated `develop` (pip install -e .)
        # re-invokes the build with isolation and loses torch.
        self._run_build(pip + ["install", ".", "--no-build-isolation", "--no-deps"],
                        cr, env, "install custom_rasterizer")

        dr = vendor / "DifferentiableRenderer"
        self._write_inpaint_setup(dr)
        self._run_build([sys.executable, "_build_inpaint.py", "build_ext", "--inplace"],
                        dr, env, "compile mesh_inpaint_processor")
        print("[Hunyuan3D21Generator] Native PBR extensions built.")

    def _patch_native_sources(self, vendor: Path) -> None:
        """Apply the MSVC/runtime portability patches idempotently."""
        for rel, repls in _NATIVE_PATCHES.items():
            path = vendor / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            new = text
            for old, repl in repls:
                if old in new:
                    new = new.replace(old, repl)
            if new != text:
                path.write_text(new, encoding="utf-8")
                print(f"[Hunyuan3D21Generator] Patched {rel}")

    def _write_inpaint_setup(self, dr: Path) -> None:
        (dr / "_build_inpaint.py").write_text(
            "from setuptools import setup\n"
            "from pybind11.setup_helpers import Pybind11Extension, build_ext\n"
            "setup(name='mesh_inpaint_processor', cmdclass={'build_ext': build_ext},\n"
            "      ext_modules=[Pybind11Extension('mesh_inpaint_processor',\n"
            "          ['mesh_inpaint_processor.cpp'], cxx_std=17)])\n",
            encoding="utf-8",
        )

    def _run_build(self, cmd, cwd: Path, env: dict, label: str) -> None:
        import subprocess
        print(f"[Hunyuan3D21Generator] {label}…")
        proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-25:])
            raise RuntimeError(f"Build step '{label}' failed (exit {proc.returncode}):\n{tail}")

    def _build_env(self) -> dict:
        """Environment for compiling CUDA/C++ extensions. On Windows this loads
        the MSVC x64 toolchain and CUDA Toolkit; elsewhere it assumes the
        compilers are already on PATH."""
        import subprocess
        env = dict(os.environ)
        if sys.platform != "win32":
            return env

        vcvars = self._find_vcvars()
        cuda = self._find_cuda()
        if vcvars is None or cuda is None:
            raise RuntimeError(self._prereq_message(vcvars, cuda))

        out = subprocess.run(f'"{vcvars}" >nul 2>&1 && set', shell=True,
                             capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
        env["CUDA_PATH"] = str(cuda)
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
        env["DISTUTILS_USE_SDK"] = "1"  # required by torch when VC env is pre-activated
        return env

    @staticmethod
    def _find_vcvars():
        import subprocess
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.exists():
            return None
        try:
            out = subprocess.check_output(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"], text=True).strip()
        except Exception:
            return None
        if not out:
            return None
        vcvars = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        return vcvars if vcvars.exists() else None

    @staticmethod
    def _find_cuda():
        cp = os.environ.get("CUDA_PATH")
        if cp and (Path(cp) / "bin" / "nvcc.exe").exists():
            return Path(cp)
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands = sorted((d for d in base.iterdir() if (d / "bin" / "nvcc.exe").exists()),
                           reverse=True)
            if cands:
                return cands[0]
        return None

    @staticmethod
    def _prereq_message(vcvars, cuda) -> str:
        missing = []
        if vcvars is None:
            missing.append("Visual Studio Build Tools 2022 (Desktop C++ / cl.exe)")
        if cuda is None:
            missing.append("CUDA Toolkit 12.x (nvcc)")
        return (
            "Cannot build the PBR native extensions — missing: " + ", ".join(missing) + ".\n"
            "Install them from an elevated terminal, open a new shell, then retry:\n\n"
            '  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override '
            '"--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"\n'
            "  winget install --id Nvidia.CUDA -e --version 12.8\n\n"
            "Shape generation works without them."
        )

    def _ensure_paint_weights(self) -> None:
        paint_dir = self.model_dir / _PAINT_SUBFOLDER
        if paint_dir.exists() and any(paint_dir.iterdir()):
            return

        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3D21Generator] Downloading PBR paint weights ({_HF_REPO_ID}/{_PAINT_SUBFOLDER})…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            allow_patterns=[f"{_PAINT_SUBFOLDER}/**"],
        )
        print("[Hunyuan3D21Generator] PBR paint weights downloaded.")

    def _ensure_realesrgan(self) -> None:
        ckpt = self.model_dir / _VENDOR_DIR_NAME / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
        if ckpt.exists():
            return
        import urllib.request
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D21Generator] Downloading RealESRGAN_x4plus.pth…")
        with urllib.request.urlopen(_REALESRGAN_URL, timeout=300) as resp:
            data = resp.read()
        ckpt.write_bytes(data)
        print(f"[Hunyuan3D21Generator] RealESRGAN saved to {ckpt}.")

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3D21Generator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3D21Generator] Downloading {_HF_REPO_ID} (shape variant)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            allow_patterns=[f"{_SHAPE_SUBFOLDER}/**"],
            ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes", "assets/**"],
        )
        print("[Hunyuan3D21Generator] Download complete.")

    def _ensure_hy3d21(self) -> None:
        """Make `hy3dshape` (and `hy3dpaint`) importable, vendoring them from the
        2.1 GitHub repo on first use."""
        vendor = self.model_dir / _VENDOR_DIR_NAME
        shape_pkg = vendor / "hy3dshape"
        paint_pkg = vendor / "hy3dpaint"

        if not shape_pkg.exists() or not paint_pkg.exists():
            self._download_repo_source(vendor)

        for p in (shape_pkg, paint_pkg):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        # `textureGenPipeline` and configs live at the hy3dpaint root.
        if str(paint_pkg) not in sys.path:
            sys.path.insert(0, str(paint_pkg))

        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dshape still not importable after extraction to {vendor}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_repo_source(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D21Generator] Downloading Hunyuan3D-2.1 source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=300) as resp:
            data = resp.read()
        print("[Hunyuan3D21Generator] Extracting hy3dshape + hy3dpaint…")

        strip = "Hunyuan3D-2.1-main/"
        wanted = (
            "Hunyuan3D-2.1-main/hy3dshape/",
            "Hunyuan3D-2.1-main/hy3dpaint/",
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(wanted):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3D21Generator] Source extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast"},
                    {"value": 30, "label": "Balanced"},
                    {"value": 50, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
                "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail but slower and more VRAM.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id":      "enable_texture",
                "label":   "Generate PBR Texture",
                "type":    "select",
                "default": False,
                "options": [
                    {"value": False, "label": "False"},
                    {"value": True,  "label": "True"},
                ],
                "tooltip": "Generate physically-based (PBR) materials. Needs ~21 GB VRAM and the compiled texture extensions.",
            },
            {
                "id":      "texture_num_view",
                "label":   "Texture Views",
                "type":    "select",
                "default": 6,
                "options": [
                    {"value": 6, "label": "6 (faster)"},
                    {"value": 8, "label": "8"},
                    {"value": 9, "label": "9 (best)"},
                ],
                "tooltip": "Number of multi-view renders used for PBR texture synthesis.",
            },
            {
                "id":      "texture_resolution",
                "label":   "Texture Resolution",
                "type":    "select",
                "default": 512,
                "options": [
                    {"value": 512, "label": "512"},
                    {"value": 768, "label": "768"},
                ],
                "tooltip": "Multi-view render resolution for PBR texture synthesis.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     0,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility. Click shuffle for a random seed.",
            },
        ]
