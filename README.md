# ZPixal3D Local Launcher + ComfyUI Node

Local launcher and ComfyUI bridge node for running **TencentARC/Pixal3D** from an isolated Python environment.

This setup keeps Pixal3D separate from the main ComfyUI environment, avoiding dependency conflicts with Torch, CUDA, custom CUDA extensions, `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `natten`, and related packages.

---

## Official projects and required third-party links

Main upstream projects:

- TencentARC/Pixal3D  
  https://github.com/TencentARC/Pixal3D

- Microsoft TRELLIS.2  
  https://github.com/microsoft/TRELLIS.2

- TRELLIS.2 project page  
  https://microsoft.github.io/TRELLIS.2/

Useful dependency / related projects:

- NATTEN  
  https://github.com/SHI-Labs/NATTEN

- NVIDIA nvdiffrast  
  https://github.com/NVlabs/nvdiffrast  
  https://nvlabs.github.io/nvdiffrast/

- Hugging Face Hub  
  https://huggingface.co/docs/huggingface_hub

- TencentARC/Pixal3D Hugging Face model page  
  https://huggingface.co/TencentARC/Pixal3D

Additional TRELLIS.2-related native extensions used by this setup:

- `CuMesh`
- `FlexGEMM`
- `o-voxel`
- `nvdiffrast`

Depending on the upstream state, these may be installed through TRELLIS.2 setup scripts, direct GitHub clones, or local source builds.

---

## Tested environment

This setup was tested on:

```text
OS: Ubuntu Server / Linux
GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM
RAM: 64 GB
Python: 3.11.14
Torch: 2.5.1+cu121
CUDA reported by Torch: 12.1
ComfyUI: separate existing ComfyUI environment
Pixal3D: dedicated external venv
```

Example environment check:

```bash
<PIXAL3D_PYTHON> -c "import sys; print(sys.executable); import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected example output:

```text
/home/user/Pixal3D/venv311/bin/python
2.5.1+cu121
True
NVIDIA GeForce RTX 3090
```

Notes:

- Linux is strongly recommended.
- NVIDIA GPU is required.
- 24 GB VRAM is recommended for comfortable use.
- `1024` resolution is the safest starting point.
- `1536` resolution can use significantly more VRAM.
- This guide was not validated on Windows or macOS.

---

## What this project provides

This repository adds two things around the official Pixal3D project:

```text
zpixal3d_official.py
```

A local command-line launcher/wrapper for Pixal3D.

```text
ComfyUI-Zetalvx-ZPixal3D/
```

A ComfyUI bridge node that sends an image to the external Pixal3D environment and returns the generated `.glb` paths.

The official TencentARC/Pixal3D source code should remain unchanged.

---

## Recommended folder structure

Example structure:

```text
<PIXAL3D_WORKSPACE>/
├── venv311/
└── ZPixal3D-Local/
    ├── Pixal3D/
    │   ├── inference.py
    │   ├── pipeline.json
    │   ├── ckpts/
    │   └── pixal3d/
    ├── TRELLIS2/
    ├── CuMesh/
    ├── FlexGEMM/
    ├── nvdiffrast/
    ├── input/
    ├── output/
    ├── comfyui_input/
    ├── comfyui_output/
    ├── logs/
    └── zpixal3d_official.py
```

ComfyUI node structure:

```text
<COMFYUI_ROOT>/
└── custom_nodes/
    └── ComfyUI-Zetalvx-ZPixal3D/
        ├── __init__.py
        └── zpixal3d_node.py
```

Example placeholders used in this guide:

```text
<PIXAL3D_WORKSPACE> = /home/user/Pixal3D
<ZPIXAL3D_ROOT>     = /home/user/Pixal3D/ZPixal3D-Local
<PIXAL3D_REPO>      = /home/user/Pixal3D/ZPixal3D-Local/Pixal3D
<PIXAL3D_PYTHON>    = /home/user/Pixal3D/venv311/bin/python
<COMFYUI_ROOT>      = /home/user/ai/ComfyUI
```

Replace these paths with your own local paths.

---

## Dependency files to include in this project

Recommended files to include in your GitHub repository:

```text
README.md
zpixal3d_official.py
requirements-zpixal3d-extra.txt
ComfyUI-Zetalvx-ZPixal3D/
├── __init__.py
└── zpixal3d_node.py
```

Suggested optional helper files:

```text
scripts/
├── 01_clone_pixal3d.sh
└── 02_check_env.sh
```

The `scripts/` folder is optional. It is useful for checking the environment or cloning the upstream Pixal3D repository, but it is not required at runtime.

---

## Suggested `requirements-zpixal3d-extra.txt`

This project does **not** try to replace the official Pixal3D/TRELLIS.2 installation.

Use the official installation instructions first, then install only missing extras.

Suggested extra requirements file:

```text
trimesh
opencv-python-headless
huggingface_hub
pillow
numpy
tqdm
```

Install with:

```bash
<PIXAL3D_PYTHON> -m pip install -r requirements-zpixal3d-extra.txt
```

Important:

- Do not blindly reinstall Torch/CUDA if your environment already works.
- Check Torch first.
- Keep ComfyUI and Pixal3D in separate environments.

Check Torch:

```bash
<PIXAL3D_PYTHON> -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## Native dependencies / compiled modules

Pixal3D/TRELLIS.2 may require several native or CUDA-backed components:

```text
cumesh
flex_gemm
o_voxel
nvdiffrast
natten
```

Recommended check:

```bash
<PIXAL3D_PYTHON> - <<'PY'
mods = [
    "torch",
    "cv2",
    "natten",
    "trimesh",
    "huggingface_hub",
    "PIL",
    "cumesh",
    "flex_gemm",
    "o_voxel",
    "nvdiffrast",
]
for mod in mods:
    try:
        __import__(mod)
        print(f"{mod}: OK")
    except Exception as e:
        print(f"{mod}: missing/problem -> {e}")
PY
```

If one of these modules is missing, follow the upstream Pixal3D/TRELLIS.2 installation instructions or build the missing native package locally.

---
## Important note about Pixal3D source code

Do **not** patch the official Pixal3D Python source code.

Files such as these should remain original:

```text
pixal3d/models/sparse_structure_flow.py
pixal3d/models/structured_latent_flow.py
pixal3d/modules/utils.py
```

If you get errors like:

```text
'dict' object has no attribute 'shape'
'dict' object has no attribute 'type'
linear input must be Tensor, not dict
```

the issue is usually not the Python model code. It is often caused by incomplete checkpoint JSON configuration.

The fix is to patch the JSON files in:

```text
<PIXAL3D_REPO>/ckpts/
```

---

## Checkpoint JSON fix

Some local/manual checkpoint downloads may miss the `proj` conditioning configuration.

The relevant field is:

```json
"image_attn_mode": "proj"
```

For SLat models, also:

```json
"proj_in_channels": 2048,
"dtype": "bfloat16"
```

Patch the JSON files:

```bash
cd <PIXAL3D_REPO>

python - <<'PY'
import json
from pathlib import Path

patches = {
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json": {
        "image_attn_mode": "proj",
        "dtype": "bfloat16",
    },
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json": {
        "image_attn_mode": "proj",
        "proj_in_channels": 2048,
        "dtype": "bfloat16",
    },
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json": {
        "image_attn_mode": "proj",
        "proj_in_channels": 2048,
        "dtype": "bfloat16",
    },
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json": {
        "image_attn_mode": "proj",
        "proj_in_channels": 2048,
        "dtype": "bfloat16",
    },
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json": {
        "image_attn_mode": "proj",
        "proj_in_channels": 2048,
        "dtype": "bfloat16",
    },
}

for rel, values in patches.items():
    path = Path(rel)
    if not path.exists():
        print(f"SKIP missing: {path}")
        continue

    data = json.loads(path.read_text())
    args = data.setdefault("args", {})
    args.update(values)
    path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"PATCHED: {path}")

print("DONE")
PY
```

Verify:

```bash
grep -R "image_attn_mode\|proj_in_channels\|dtype" -n ckpts/*.json
```

---

## Files where paths must be edited

There are only two project files where users normally need to edit paths.

### 1. `zpixal3d_official.py`

Location:

```text
<ZPIXAL3D_ROOT>/zpixal3d_official.py
```

Find:

```python
python_for_inference = "/home/user/Pixal3D/venv311/bin/python"
```

Change it to your Pixal3D venv Python path:

```python
python_for_inference = "<PIXAL3D_PYTHON>"
```

Example:

```python
python_for_inference = "/home/user/Pixal3D/venv311/bin/python"
```

This matters because the wrapper may be launched by another environment, such as ComfyUI. Pixal3D inference must still run inside the Pixal3D venv.

### 2. `zpixal3d_node.py`

Location:

```text
<COMFYUI_ROOT>/custom_nodes/ComfyUI-Zetalvx-ZPixal3D/zpixal3d_node.py
```

Check the default paths:

```python
"default": "/home/user/Pixal3D/venv311/bin/python"
```

```python
"default": "/home/user/Pixal3D/ZPixal3D-Local"
```

```python
"default": "/home/user/Pixal3D/ZPixal3D-Local/Pixal3D"
```

Also check this safety line:

```python
python_bin = Path("/home/user/Pixal3D/venv311/bin/python").resolve()
```

Change these paths to match your installation.

---

## Terminal usage

### Single image

```bash
cd <ZPIXAL3D_ROOT>

<PIXAL3D_PYTHON> zpixal3d_official.py \
  --path input/test.png \
  --output output \
  --model-path <PIXAL3D_REPO> \
  --resolution 1024 \
  --low-vram \
  --extract-textures
```

### Batch mode

If `--path` is omitted, the script processes all supported images inside:

```text
<ZPIXAL3D_ROOT>/input/
```

Run:

```bash
cd <ZPIXAL3D_ROOT>

<PIXAL3D_PYTHON> zpixal3d_official.py \
  --output output \
  --model-path <PIXAL3D_REPO> \
  --resolution 1024 \
  --low-vram \
  --delay 20 \
  --extract-textures
```

Supported image formats:

```text
.png
.jpg
.jpeg
.webp
.bmp
.tif
.tiff
```

---

## Output structure

For each input image, the wrapper creates a dedicated folder:

```text
output/
└── image_name/
    ├── image_name_input.png
    ├── image_name.glb
    ├── run.log
    └── textures/
```

Texture extraction is best-effort. Some GLB files store textures internally. In that case, separate PNG texture files may not always be extracted even when the GLB itself is correctly textured.

---

## Resolution

Supported values:

```text
1024
1536
```

Examples:

```bash
--resolution 1024
```

```bash
--resolution 1536
```

`1536` uses more VRAM and can fail on heavier images. Start with `1024`.

---
## ComfyUI usage

Install the node here:

```text
<COMFYUI_ROOT>/custom_nodes/ComfyUI-Zetalvx-ZPixal3D/
```

Required files:

```text
__init__.py
zpixal3d_node.py
```

Restart ComfyUI after installation:

```bash
cd <COMFYUI_ROOT>
source venv/bin/activate
python main.py
```

The node appears under:

```text
ZETALVX/ZPixal3D
```

Node name:

```text
ZPixal3D Image to 3D
```

The node:

```text
1. receives an IMAGE from ComfyUI
2. saves it to <ZPIXAL3D_ROOT>/comfyui_input/
3. launches zpixal3d_official.py using the external Pixal3D venv
4. generates a GLB
5. copies the GLB into <COMFYUI_ROOT>/input/3d/
6. returns useful output paths
```

Returned outputs:

```text
glb_path
comfyui_3d_path
output_folder
log_path
```

The copied file inside:

```text
<COMFYUI_ROOT>/input/3d/
```

can be loaded by ComfyUI 3D nodes such as `Load 3D`.

---

## VRAM / OOM handling

The wrapper runs Pixal3D inference in a subprocess. This helps release VRAM more reliably between images.

Useful options:

```bash
--low-vram
--delay 20
```

The script sets:

```text
ATTN_BACKEND=sdpa
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For batch processing, use a delay between images:

```bash
--delay 30
```

For `1536` resolution, a longer delay may help:

```bash
--resolution 1536 --delay 60
```

---

## Optional texture extraction

To enable texture extraction attempt, install `trimesh` in the Pixal3D venv:

```bash
<PIXAL3D_PYTHON> -m pip install trimesh --no-deps
```

Then run with:

```bash
--extract-textures
```

Note: GLB files may contain embedded textures/materials. Separate texture PNG files are not guaranteed.

---

## Environment check

Check the Pixal3D venv:

```bash
<PIXAL3D_PYTHON> -c "import sys; print(sys.executable); import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected example:

```text
<PIXAL3D_PYTHON>
2.5.1+cu121
True
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Pixal3D is being launched with the wrong Python.

Check:

```bash
<PIXAL3D_PYTHON> -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Then verify these files:

```text
<ZPIXAL3D_ROOT>/zpixal3d_official.py
<COMFYUI_ROOT>/custom_nodes/ComfyUI-Zetalvx-ZPixal3D/zpixal3d_node.py
```

They must point to:

```text
<PIXAL3D_PYTHON>
```

not to:

```text
/usr/bin/python3
/usr/bin/python3.11
```

### GLB generated but no separate textures

This can be normal. Pixal3D may save textures embedded in the `.glb`.

If you want to try extracting separate PNG textures:

```bash
<PIXAL3D_PYTHON> -m pip install trimesh --no-deps
```

Then run with:

```bash
--extract-textures
```

### Conditioning errors

If you see:

```text
'dict' object has no attribute 'shape'
'dict' object has no attribute 'type'
linear input must be Tensor, not dict
```

do not patch Pixal3D Python code.

Patch the checkpoint JSON files in:

```text
<PIXAL3D_REPO>/ckpts/
```

using the JSON patch described above.

---

## Summary

This setup keeps the official Pixal3D source code unchanged.

The project adds:

```text
zpixal3d_official.py
```

as a local launcher/wrapper, and:

```text
ComfyUI-Zetalvx-ZPixal3D/
```

as a ComfyUI bridge node.

The only Pixal3D-side correction that may be needed is the checkpoint JSON configuration for:

```json
"image_attn_mode": "proj"
```

and the related SLat settings:

```json
"proj_in_channels": 2048,
"dtype": "bfloat16"
```
