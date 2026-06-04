import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


class ZPixal3DImageTo3D:
    """
    ComfyUI bridge node for ZPixal3D.

    ComfyUI remains in its own Python environment.
    ZPixal3D is launched through its external venv:

        /home/zetalvx/Pixal3D/venv311/bin/python
        /home/zetalvx/Pixal3D/ZPixal3D-Local/zpixal3d_official.py

    The node saves the input image, runs ZPixal3D, copies the generated GLB
    into ComfyUI/input/3d, and returns useful paths as strings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "output_name": (
                    "STRING",
                    {
                        "default": "zpixal3d_output",
                        "multiline": False,
                    },
                ),

                "resolution": (
                    ["1024", "1536"],
                    {
                        "default": "1024",
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2147483647,
                    },
                ),

                "delay": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 1.0,
                    },
                ),

                "low_vram": (
                    ["true", "false"],
                    {
                        "default": "true",
                    },
                ),

                "extract_textures": (
                    ["false", "true"],
                    {
                        "default": "true",
                    },
                ),

                "python_path": (
                    "STRING",
                    {
                        "default": "/home/zetalvx/Pixal3D/venv311/bin/python",
                        "multiline": False,
                    },
                ),

                "zpixal3d_root": (
                    "STRING",
                    {
                        "default": "/home/zetalvx/Pixal3D/ZPixal3D-Local",
                        "multiline": False,
                    },
                ),

                "model_path": (
                    "STRING",
                    {
                        "default": "/home/zetalvx/Pixal3D/ZPixal3D-Local/Pixal3D",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "glb_path",
        "comfyui_3d_path",
        "output_folder",
        "log_path",
    )

    FUNCTION = "run"
    CATEGORY = "ZETALVX/ZPixal3D"
    OUTPUT_NODE = True

    def _safe_name(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            name = f"zpixal3d_{time.strftime('%Y%m%d_%H%M%S')}"

        bad = '<>:"/\\|?* '
        for ch in bad:
            name = name.replace(ch, "_")

        while "__" in name:
            name = name.replace("__", "_")

        return name.strip("_") or f"zpixal3d_{time.strftime('%Y%m%d_%H%M%S')}"

    def _save_image(self, image_tensor, mask_tensor, save_path: Path, index: int):
        """
        ComfyUI IMAGE tensor is usually [B,H,W,C], float 0..1.
        MASK is usually [B,H,W], float 0..1.
        """

        img = image_tensor[index].detach().cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        if img.shape[-1] == 4:
            pil = Image.fromarray(img, mode="RGBA")
        else:
            pil = Image.fromarray(img[:, :, :3], mode="RGB")

        if mask_tensor is not None:
            mask = mask_tensor[index].detach().cpu().numpy()
            mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)

            if pil.mode != "RGBA":
                pil = pil.convert("RGBA")

            pil.putalpha(Image.fromarray(mask, mode="L"))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(save_path)

    def _read_tail(self, path: Path, max_chars: int = 7000) -> str:
        try:
            if path and path.exists() and path.is_file():
                return path.read_text(errors="ignore")[-max_chars:]
        except Exception:
            pass
        return ""

    def _find_recent_glb(self, output_root: Path, start_time: float):
        candidates = []

        if output_root.exists():
            for p in output_root.rglob("*.glb"):
                try:
                    if p.is_file() and p.stat().st_mtime >= start_time - 2:
                        candidates.append(p)
                except Exception:
                    pass

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _find_latest_log(self, output_root: Path):
        candidates = []

        if output_root.exists():
            for p in output_root.rglob("run.log"):
                try:
                    if p.is_file():
                        candidates.append(p)
                except Exception:
                    pass

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _get_comfy_root(self) -> Path:
        """
        This file is inside:
            ComfyUI/custom_nodes/ZPixal3D-ComfyUI-Node/zpixal3d_node.py

        parents[2] should be the ComfyUI root.
        """
        return Path(__file__).resolve().parents[2]

    def _copy_to_comfyui_3d_input(self, glb_path: Path, preferred_name: str) -> Path:
        """
        Copy generated GLB into ComfyUI/input/3d so native 3D nodes can load it.
        """
        comfy_root = self._get_comfy_root()
        target_dir = comfy_root / "input" / "3d"
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_name(preferred_name)
        target = target_dir / f"{safe_name}.glb"

        if target.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target = target_dir / f"{safe_name}_{stamp}.glb"

        shutil.copy2(glb_path, target)
        return target

    def _run_external(
        self,
        python_path: str,
        zpixal3d_root: str,
        model_path: str,
        input_png: Path,
        output_root: Path,
        resolution: str,
        seed: int,
        delay: float,
        low_vram: str,
        extract_textures: str,
    ):
        root = Path(os.path.expanduser(zpixal3d_root)).resolve()
        # Force external ZPixal3D venv.
        # Do not trust the workflow value here, because old workflows/widgets can pass system Python.
        python_bin = Path("/home/zetalvx/Pixal3D/venv311/bin/python").resolve()
        model = Path(os.path.expanduser(model_path)).resolve()
        script = root / "zpixal3d_official.py"

        # Safety fallback: never use system Python for Pixal3D.
        if str(python_bin) in ["/usr/bin/python3.11", "/usr/bin/python3", "/usr/bin/python"]:
            python_bin = Path("/home/zetalvx/Pixal3D/venv311/bin/python").resolve()

        if not python_bin.exists():
            raise RuntimeError(f"ZPixal3D Python venv not found:\n{python_bin}")

        if not script.exists():
            raise RuntimeError(f"zpixal3d_official.py not found:\n{script}")

        if not model.exists():
            raise RuntimeError(f"Model path not found:\n{model}")

        output_root.mkdir(parents=True, exist_ok=True)

        stem = input_png.stem
        expected_item_dir = output_root / stem
        expected_glb_path = expected_item_dir / f"{stem}.glb"
        expected_log_path = expected_item_dir / "run.log"
        node_log_path = output_root / f"{stem}_comfyui_node.log"

        # Clean only previous expected output.
        if expected_glb_path.exists():
            if expected_glb_path.is_dir():
                shutil.rmtree(expected_glb_path)
            else:
                expected_glb_path.unlink()

        cmd = [
            "/home/zetalvx/Pixal3D/venv311/bin/python",
            str(script),
            "--path",
            str(input_png),
            "--output",
            str(output_root),
            "--model-path",
            str(model),
            "--resolution",
            str(resolution),
            "--seed",
            str(seed),
            "--delay",
            str(delay),
        ]

        if low_vram == "false":
            cmd.append("--no-low-vram")

        if extract_textures == "true":
            cmd.append("--extract-textures")

        env = os.environ.copy()
        env["ATTN_BACKEND"] = "sdpa"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Important: force child inference to use external Pixal3D venv.
        env["ZPIXAL3D_PYTHON"] = str(python_bin)

        start_time = time.time()

        header = (
            "ZPixal3D ComfyUI Node\n"
            f"START_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"PYTHON: {python_bin}\n"
            f"ROOT: {root}\n"
            f"SCRIPT: {script}\n"
            f"MODEL: {model}\n"
            f"INPUT: {input_png}\n"
            f"OUTPUT_ROOT: {output_root}\n"
            f"EXPECTED_GLB: {expected_glb_path}\n"
            f"COMMAND: {' '.join(cmd)}\n\n"
        )

        print(header)

        stdout_lines = []

        with node_log_path.open("w", encoding="utf-8") as node_log:
            node_log.write(header)
            node_log.flush()

            process = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    node_log.write(line)
                    node_log.flush()

                    stdout_lines.append(line)
                    if len(stdout_lines) > 700:
                        stdout_lines = stdout_lines[-700:]

            return_code = process.wait()

        stdout_tail = "".join(stdout_lines)[-9000:]

        if return_code != 0:
            latest_log = self._find_latest_log(output_root)
            latest_log_tail = self._read_tail(latest_log) if latest_log else ""

            raise RuntimeError(
                "ZPixal3D external script failed.\n\n"
                f"Return code: {return_code}\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"Node log:\n{node_log_path}\n\n"
                f"External stdout tail:\n{stdout_tail}\n\n"
                f"Latest run.log:\n{latest_log}\n\n"
                f"Latest run.log tail:\n{latest_log_tail}"
            )

        # Preferred expected path.
        if expected_glb_path.exists() and expected_glb_path.is_file():
            return expected_glb_path, expected_item_dir, expected_log_path

        # Fallback: search any GLB created after this run started.
        found_glb = self._find_recent_glb(output_root, start_time)
        if found_glb is not None:
            item_dir = found_glb.parent
            log_path = item_dir / "run.log"
            return found_glb, item_dir, log_path

        latest_log = self._find_latest_log(output_root)
        latest_log_tail = self._read_tail(latest_log) if latest_log else ""

        raise RuntimeError(
            "ZPixal3D did not create the GLB.\n\n"
            f"Expected:\n{expected_glb_path}\n\n"
            f"Searched in:\n{output_root}\n\n"
            f"Node log:\n{node_log_path}\n\n"
            f"External stdout tail:\n{stdout_tail}\n\n"
            f"Latest run.log:\n{latest_log}\n\n"
            f"Latest run.log tail:\n{latest_log_tail}"
        )

    def run(
        self,
        image,
        output_name,
        resolution,
        seed,
        delay,
        low_vram,
        extract_textures,
        python_path,
        zpixal3d_root,
        model_path,
        mask=None,
    ):
        root = Path(os.path.expanduser(zpixal3d_root)).resolve()

        comfy_input_dir = root / "comfyui_input"
        comfy_output_dir = root / "comfyui_output"

        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        comfy_output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self._safe_name(output_name)

        batch_size = image.shape[0]

        glb_paths = []
        comfyui_3d_paths = []
        output_folders = []
        log_paths = []

        for i in range(batch_size):
            if batch_size > 1:
                name = f"{base_name}_{i:03d}"
            else:
                name = base_name

            input_png = comfy_input_dir / f"{name}.png"

            self._save_image(
                image_tensor=image,
                mask_tensor=mask,
                save_path=input_png,
                index=i,
            )

            glb_path, item_dir, log_path = self._run_external(
                python_path=python_path,
                zpixal3d_root=zpixal3d_root,
                model_path=model_path,
                input_png=input_png,
                output_root=comfy_output_dir,
                resolution=resolution,
                seed=seed,
                delay=delay,
                low_vram=low_vram,
                extract_textures=extract_textures,
            )

            comfyui_3d_path = self._copy_to_comfyui_3d_input(
                glb_path=glb_path,
                preferred_name=name,
            )

            glb_paths.append(str(glb_path))
            comfyui_3d_paths.append(str(comfyui_3d_path))
            output_folders.append(str(item_dir))
            log_paths.append(str(log_path))

        return (
            "\n".join(glb_paths),
            "\n".join(comfyui_3d_paths),
            "\n".join(output_folders),
            "\n".join(log_paths),
        )


NODE_CLASS_MAPPINGS = {
    "ZPixal3DImageTo3D": ZPixal3DImageTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZPixal3DImageTo3D": "ZPixal3D Image to 3D",
}