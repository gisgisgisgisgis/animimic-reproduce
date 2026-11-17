# AniMimic reproduction

Research implementation of AniMimic using existing rigged meshes:

```text
GLB → canonical render → cached Wan / LTX I2V → SAM2 / AllTracker / VGGT
    → joint optimization → fTetWild → differentiable FEM → NPZ / GLB / MP4
```

All stages are connected by `animimic run`. Numerical tests run on CPU.
The complete pretrained-model pipeline has **not been executed on a 4090** in
this environment, which has no accessible NVIDIA GPU. Generated benchmark
quality, runtime and peak GPU memory are therefore unverified.

For lighter local generation, select **LTX-Video 2B** with
`animimic run --config configs/horse-ltx.yaml`. Wan remains available unchanged.
See [本地影片模型切換、八個資產與實際預覽](docs/local-video.md) for setup,
standalone video generation and the five additional textured animal models.

Use Python 3.11 or 3.12 in an isolated virtual environment. Install a CUDA
PyTorch build compatible with the target machine first:

```bash
pip install -e '.[gpu]'
python scripts/bootstrap.py
# Build third_party/fTetWild with CMake and add FloatTetwild_bin to PATH.
# Download Wan-AI/Wan2.2-TI2V-5B to weights/Wan2.2-TI2V-5B.
```

PyTorch 2.6 / torchvision 0.21 are pinned because the newest PyTorch build
tested here failed to compile PyTorch3D. CPU numerical versions are recorded
in `requirements-tested.txt`.

The bootstrap script clones official inference repositories, installs their
dependencies, and records their Git revisions in `third_party/revisions.json`.
Upstream CUDA build dependencies may require machine-specific installation.

```bash
animimic assets list
animimic assets fetch --asset fox
animimic inspect assets/downloads/fox/model.glb
animimic preflight --config configs/fox.yaml
animimic run --config configs/fox.yaml
animimic run --config configs/fox.yaml --from-stage joints
animimic run --config configs/fox.yaml --force-stage supervision
PYTHONPATH=src python -m unittest discover -s tests -v
```

Use `configs/humanoid.yaml` or `configs/robot.yaml` for the original debug benchmarks.
The expanded set also includes horse, puppy, shark, seagull and sea turtle,
with individual `configs/<asset>-ltx.yaml` presets.
Set `asset_path` to use a local rigged GLB; `asset` remains its run label.
Set `reference_video` to reuse an external video instead of generating one.
Set `device: cpu` for small numerical experiments with a CPU PyTorch3D build.

Each stage writes a content-addressed artifact directory. The run's
`manifest.json` records paths and SHA-256 hashes. A corrupted cached file causes
that stage to rerun. Video keys exclude joint/FEM settings, so tuning those
settings reuses reference videos. Changing prompt, seed or canonical image
creates a new video key. Force flags intentionally overwrite a stage's cached
artifacts; keep the original manifest/video if comparing stochastic reruns.

Original debug assets (downloaded individually; manifests retain hashes and credits):

| Asset | Source | License |
| --- | --- | --- |
| Fox | [Khronos](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox) | CC BY 4.0; PixelMannen, tomkranis, AsoboStudio, scurest |
| CesiumMan | [Khronos](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan) | CC BY 4.0; Cesium; logo trademark reserved |
| RobotExpressive | [three.js](https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/RobotExpressive) | CC0; Tomas Laulhe / Quaternius, Don McCurdy |

Bake3D downloads returned HTTP 403 during implementation, so the benchmark uses
these public alternatives. Downloaded assets, weights and runs are ignored by Git.
The five naturalistic animal additions are individually fetched from Objaverse,
with original source/CC BY attribution and pinned hashes in
`src/animimic/animal_assets.json`. Their previews and face/joint counts are in
[the asset guide](docs/local-video.md). Fox is explicitly a low-poly debug model.

Joint fitting uses normalized mask, tracking, depth, RGB and temporal losses,
6D rotations, local translations, and transformation bounds. Three phases fit
root motion, geometry cues, then all losses. Joint checkpoints and loss histories
are saved between phases. Depth inference uses individual frames and exact
VGGT padding inversion; this is a reproduction choice for moving foregrounds.

FEM uses fixed-corotated elasticity, backward Euler, Newton/CG with line search,
and joint-driven boundary conditions. Material optimization updates log Young's
modulus in joint clusters, then spatially splits each cluster. Warp evaluates
the energy; PyTorch supplies the mathematically identical analytic derivatives
and matrix-free Hessian-vector products. An implicit adjoint differentiates
converged simulation states without retaining Newton iterations. Set
`physics.backend: torch` for the pure PyTorch reference backend.

The elastic energy, tangent, material gradient, rest state, synthetic motion
recovery and synthetic material fitting have automated numerical tests. No
gravity or collision/contact is modeled, following the paper's supplementary
internal-elastic-response setup. Large-deformation convergence and arbitrary
asset tetrahedralization remain benchmark-dependent.

The paper does not disclose most hyperparameters. Defaults in this code
(loss weights 1, Adam 1e-3, 25 sampled frames, Poisson ratio 0.45, density 1,
four substeps) are reproduction settings. Meshes are normalized by bounding-box
diagonal, so material units are normalized too. The present implementation
fixes Poisson ratio at 0.45.

Outputs include joint/material NPZ checkpoints, trajectories, JSON/CSV losses, skeletal
GLB, and input-view/±45-degree MP4 previews. The skeletal GLB contains joint
motion; FEM surface deformation is preserved separately in NPZ. Textures are
sampled to vertex colors for the optimization/render/export path. The pipeline
currently fits the original surface and uses a decimated proxy only for
tetrahedralization. This differs from a full textured proxy-transfer pipeline.

Export also writes `metrics.json` comparing rest, joint and physics outputs
using silhouette IoU, visible-track pixel error and vertex acceleration.
These diagnostics are not substitutes for the paper's SSIM/LPIPS/VBench study.
Tetrahedralization saves volume and embedding quality diagnostics.
