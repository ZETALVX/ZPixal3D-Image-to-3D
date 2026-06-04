#!/usr/bin/env python3
"""
ZPixal3D local launcher.

Batch/single image wrapper for TencentARC/Pixal3D.

Features:
- single file mode via --path
- batch mode from ./input if --path is not provided
- per-image output folders
- separate subprocess per image to release VRAM more reliably
- delay between images
- optional texture extraction attempt from GLB
- local model path support
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = SCRIPT_DIR / "Pixal3D"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "Pixal3D"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return []

    return [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def clean_gpu_memory(delay: float = 0.0) -> None:
    try:
        import torch

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        print(f"[WARN] GPU cleanup skipped: {e}")

    if delay > 0:
        print(f"[INFO] Waiting {delay} seconds...")
        time.sleep(delay)


def is_valid_glb(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"glTF"
    except Exception:
        return False


def try_extract_textures(glb_path: Path, textures_dir: Path) -> None:
    """
    Best-effort texture extraction.
    GLB often stores textures internally; extraction depends on how trimesh reads materials.
    """
    try:
        import trimesh
        from PIL import Image

        textures_dir.mkdir(parents=True, exist_ok=True)

        scene_or_mesh = trimesh.load(str(glb_path), force="scene")

        saved = 0

        geometries = []
        if hasattr(scene_or_mesh, "geometry"):
            geometries = list(scene_or_mesh.geometry.values())
        else:
            geometries = [scene_or_mesh]

        for idx, geom in enumerate(geometries):
            visual = getattr(geom, "visual", None)
            material = getattr(visual, "material", None)

            if material is None:
                continue

            candidates = []

            for attr in [
                "image",
                "baseColorTexture",
                "metallicRoughnessTexture",
                "normalTexture",
                "emissiveTexture",
                "occlusionTexture",
            ]:
                if hasattr(material, attr):
                    val = getattr(material, attr)
                    if val is not None:
                        candidates.append((attr, val))

            for attr, tex in candidates:
                try:
                    if isinstance(tex, Image.Image):
                        out = textures_dir / f"texture_{idx}_{attr}.png"
                        tex.save(out)
                        saved += 1
                except Exception:
                    pass

        if saved:
            print(f"[OK] Extracted {saved} texture file(s) to: {textures_dir}")
        else:
            print("[INFO] No external texture images extracted. They may be embedded in the GLB.")

    except ImportError:
        print("[WARN] trimesh/PIL not available for texture extraction.")
    except Exception as e:
        print(f"[WARN] Texture extraction failed: {e}")


def run_one(
    image_path: Path,
    output_root: Path,
    repo_path: Path,
    model_path: Path,
    resolution: int,
    low_vram: bool,
    seed: int,
    fov: float | None,
    extract_textures: bool,
) -> bool:
    stem = safe_stem(image_path)
    item_dir = output_root / stem
    textures_dir = item_dir / "textures"
    item_dir.mkdir(parents=True, exist_ok=True)

    copied_input = item_dir / f"{stem}_input{image_path.suffix.lower()}"
    final_glb = item_dir / f"{stem}.glb"
    log_path = item_dir / "run.log"

    try:
        shutil.copy2(image_path, copied_input)
    except Exception as e:
        print(f"[WARN] Could not copy input image: {e}")

    inference_py = repo_path / "inference.py"
    if not inference_py.exists():
        print(f"[ERROR] Pixal3D inference.py not found: {inference_py}")
        return False

    # Remove stale output file/folder.
    if final_glb.exists():
        if final_glb.is_dir():
            shutil.rmtree(final_glb)
        else:
            final_glb.unlink()

    python_for_inference = "/home/zetalvx/Pixal3D/venv311/bin/python"

    cmd = [
        python_for_inference,
        str(inference_py),
        "--image", str(image_path),
        "--output", str(final_glb),
        "--seed", str(seed),
        "--resolution", str(resolution),
        "--model_path", str(model_path),
    ]

    if low_vram:
        cmd.append("--low_vram")

    if fov is not None:
        cmd += ["--fov", str(fov)]

    env = os.environ.copy()
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("\n" + "=" * 80)
    print(f"[INFO] INPUT     : {image_path}")
    print(f"[INFO] ITEM DIR  : {item_dir}")
    print(f"[INFO] OUTPUT GLB: {final_glb}")
    print(f"[INFO] LOG       : {log_path}")
    print(f"[INFO] WRAPPER PYTHON   : {sys.executable}")
    print(f"[INFO] INFERENCE PYTHON : {python_for_inference}")
    print(f"[INFO] CMD       : {' '.join(cmd)}")
    print(f"[INFO] ATTN_BACKEND: {env.get('ATTN_BACKEND')}")
    print("=" * 80)

    status = 1

    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(repo_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        status = process.wait()

    if status != 0:
        print(f"[ERROR] Pixal3D failed for: {image_path.name}")
        return False

    if not final_glb.exists():
        print(f"[ERROR] Process ended but GLB was not created: {final_glb}")
        return False

    if not is_valid_glb(final_glb):
        print(f"[WARN] Output exists but does not look like a valid binary GLB: {final_glb}")
    else:
        size = final_glb.stat().st_size / (1024 * 1024)
        print(f"[OK] GLB created: {final_glb} ({size:.2f} MB)")

    if extract_textures:
        try_extract_textures(final_glb, textures_dir)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZPixal3D: batch/single local image-to-3D wrapper for Pixal3D."
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Single image path. If omitted, all images in ./input are processed.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Input folder used when --path is not provided.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Main output folder.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=str(DEFAULT_REPO),
        help="Pixal3D repository folder containing inference.py.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Local Pixal3D model folder containing pipeline.json and ckpts/.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        choices=[1024, 1536],
        help="Pixal3D resolution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Optional manual FOV in radians.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=20.0,
        help="Delay in seconds between images.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=True,
        help="Use Pixal3D low VRAM mode. Enabled by default.",
    )
    parser.add_argument(
        "--no-low-vram",
        action="store_false",
        dest="low_vram",
        help="Disable low VRAM mode.",
    )
    parser.add_argument(
        "--extract-textures",
        action="store_true",
        help="Try to extract texture images from generated GLB.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    repo_path = Path(args.repo).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    if args.path:
        image = Path(args.path).expanduser()
        if not image.is_absolute():
            image = (Path.cwd() / image).resolve()
        else:
            image = image.resolve()

        if not image.exists() or not image.is_file():
            print(f"[ERROR] Image not found: {image}")
            sys.exit(1)

        images = [image]
    else:
        images = collect_images(input_dir)

    if not images:
        print(f"[INFO] No images found in: {input_dir}")
        print("[INFO] Put images into ./input or use --path /path/to/image.png")
        return

    print(f"[INFO] Images to process: {len(images)}")
    print(f"[INFO] Output root      : {output_root}")
    print(f"[INFO] Pixal3D repo     : {repo_path}")
    print(f"[INFO] Model path       : {model_path}")
    print(f"[INFO] Resolution       : {args.resolution}")
    print(f"[INFO] Low VRAM         : {args.low_vram}")
    print(f"[INFO] Delay            : {args.delay}")

    ok_count = 0
    fail_count = 0

    for index, image_path in enumerate(images, start=1):
        print(f"\n[INFO] Item {index}/{len(images)}")

        success = run_one(
            image_path=image_path,
            output_root=output_root,
            repo_path=repo_path,
            model_path=model_path,
            resolution=args.resolution,
            low_vram=args.low_vram,
            seed=args.seed,
            fov=args.fov,
            extract_textures=args.extract_textures,
        )

        if success:
            ok_count += 1
        else:
            fail_count += 1

        clean_gpu_memory(args.delay)

    print("\n" + "=" * 80)
    print("[DONE] Batch completed.")
    print(f"[OK] Failed : {fail_count}")
    print(f"[OK] Success: {ok_count}")
    print(f"[INFO] Outputs: {output_root}")
    print("=" * 80)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
