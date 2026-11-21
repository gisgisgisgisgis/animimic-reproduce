# AnimaMimic / AniMimic reproduction: technical notes

This file holds the per-script details in English. The repository front page is [README.md](../README.md) (中文). Paths below are relative to the repository root.

Independent reimplementation of *AniMimic: Imitating 3D Animation from Video Priors* (CVPR 2026, arXiv 2512.14133; the arXiv version is titled AnimaMimic). The authors have not released code.

The main path targets Linux + CUDA, which has not been run yet. The whole pipeline has been run on Apple Silicon: Stage 1 and rendering use MPS through `RasterRenderer`, FEM physics stays on CPU because MPS has no float64, and UniRig uses CPU through `scripts/prep_00_unirig_mac.sh`.

## Install on a CUDA machine

```bash
bash setup_cuda.sh
```

This creates `animamimic` and `animamimic-prep`, and clones AllTracker and UniRig into `third_party/`. Then create the AllTracker and UniRig environments from their own READMEs.

## Run one mesh

Each sequence lives in `data/<name>/`. Run the steps in order.

| Step | Env | Command | Output |
|---|---|---|---|
| Rig the mesh | UniRig | `bash scripts/prep_00_unirig.sh third_party/UniRig input.glb data/<name>` | `rigged.glb` |
| Render the canonical view | animamimic | `python scripts/prep_01_render_view.py --seq data/<name>` | `reference.png`, `reference_mask.png`, `camera.json` |
| Generate a video | external | Give `reference.png` and a motion prompt to Kling or another image-to-video model | `.mp4` |
| Import the video | animamimic | `python scripts/prep_02_import_video.py --seq data/<name> --video out.mp4` | `frames/` |
| Masks | animamimic-prep | `python scripts/prep_04_sam2.py --seq data/<name>` | `masks/` |
| Tracks | AllTracker | `python scripts/prep_05_alltracker.py --seq data/<name> --repo third_party/alltracker` | `tracks.npz` |
| Depth | animamimic-prep | `python scripts/prep_06_vggt.py --seq data/<name>` | `depth.npz` |
| Tet mesh (Stage 2) | animamimic | `python scripts/prep_07_tetmesh.py --seq data/<name>` | `tet.npz` |
| Optimize | animamimic | `python scripts/run_animamimic.py --seq data/<name> --out outputs/<name> --stage2` | `stage1.npz`, `stage2.npz`, comparison videos |

The rig script first bakes every node transform into the mesh (`scripts/prep_00a_bake_mesh.py`). Without it, UniRig's merge step can write the mesh and the new skeleton in different frames when the input mesh sits under a transformed node. The horse asset had a 126 degree yaw and 0.9 scale on its node, and its skin weights were transferred between the two misaligned frames. `prep_01_render_view.py` stops when the median vertex-to-bone distance is above 0.15 of the mesh size. To look at a rig, `python scripts/view_rig.py --seq data/<name> --out rig.png` draws the skeleton over the mesh colored by its skin weights, from the input camera, the side and −45°.

## Run the paper's own reference videos

The project page publishes 8 reference videos and the paper's input-view result for each, in `data/paper_refs/` (see its README). They come without meshes, so a mesh is rebuilt from frame 0 and placed to match it. The paper's assets were image-to-3D outputs too.

| Step | Env | Command | Output |
|---|---|---|---|
| Mesh from frame 0 | `third_party/venvs/hunyuan3d` | `python scripts/prep_09a_frame0_to_3d.py --video data/paper_refs/ref_4.mp4 --seq data/paper_dog` | `frame0.png`, `frame0_mask.png`, `shape.obj` |
| Fit the camera, texture from frame 0 | animamimic | `python scripts/prep_09b_fit_frame0.py --seq data/paper_dog` | `input.glb`, `camera.json`, `fit_frame0.png` |
| Rig | UniRig | `bash scripts/prep_00_unirig_mac.sh third_party/UniRig data/paper_dog/input.glb data/paper_dog` | `rigged.glb` |
| Reference view | animamimic | `python scripts/prep_01_render_view.py --seq data/paper_dog --keep_camera` | `reference.png` |
| Import, masks, tracks, depth, Stage 1 | | as above, with `--video data/paper_refs/ref_4.mp4` | `stage1.npz` |
| Compare with the paper | animamimic | `python tests/compare_with_paper.py --seq data/paper_dog --paper data/paper_refs/ours_4.mp4 --result outputs/paper_dog/stage1.npz --size 256 --out outputs/paper_dog/compare_with_paper` | `.json`, `.png`, `.mp4` |
| Bake a texture (looks) | animamimic | `python scripts/bake_texture.py --seq data/paper_dog` | `texture_baked.png`, `texture_baked.npz`, `textured_rest.glb` |
| Compare, sharp 512 px | animamimic | add `--texture baked --size 512 --sigma 1e-6` to the comparison | as above |
| Stage 1, sharp + rigid | animamimic | `python scripts/run_animamimic.py --seq data/paper_dog_h2_fov --out outputs/paper_dog_h2_fov_sharp_rigid --device mps --backend raster --image_size 384 --sigma 1e-5 --w_rigid 10 --iters 3000 --batch_frames 4 --depth_norm per_frame --trans_max 0.01 --w_reg 1000 --skin_smooth 30` | `stage1.npz` |
| Tet mesh, Stage 2 | animamimic | `python scripts/prep_07_tetmesh.py --seq data/paper_dog_h2_fov`, then `python scripts/run_animamimic.py --seq data/paper_dog_h2_fov --out outputs/paper_dog_h2_fov_sharp_rigid --device mps --backend raster --image_size 384 --sigma 1e-5 --batch_frames 4 --depth_norm per_frame --skin_smooth 30 --stage1_from outputs/paper_dog_h2_fov_sharp_rigid/stage1.npz --stage2 --stage2_iters 10 10 --fps 6.2` | `stage2.npz` |

`prep_09b --fit_fov` also fits the field of view, as an image-plane zoom that is converted back to a fov. With the fixed 40° the dog's camera sat too far and too narrow, the fit pushed the mesh's snout outside frame 0, and the snout's texture came out without the black nose. The fitted fov was 63.8° for the dog (fit IoU 0.848 → 0.881) and 45.4° for the wolf. To choose between candidate meshes, `python tests/rest_views.py --seq data/<name> --out outputs/mesh_choice/<name>` renders the undeformed baked mesh from the four QA views, and `frame_qa_vlm.py` scores them; the dog uses the fov-fitted full Hunyuan3D-2 mesh (VLM total 27 vs 25), the wolf keeps its first mesh (29 vs 28).

`--w_rigid` adds a term that is not in the paper: the mean squared relative change of every mesh edge length, with rest lengths floored at a quarter of the median edge so that sliver edges do not dominate. Without it, joints at the tips of the skeleton (snout, ears, paws) rotate almost freely, because few pixels constrain them; on the dog the snout-tip joint averaged 50° and an ear tip reached 178°, which creased the face and curled the ears. With `--w_rigid 10` the creases and curled ears are gone, and the input-view metrics barely change (dog SSIM 0.918 → 0.917) or improve (wolf SSIM 0.909 → 0.913, LPIPS 0.084 → 0.077, IoU 0.894 → 0.916). `--sigma 1e-5 --image_size 384` renders sharper silhouettes during optimization than the default 1e-4 at 256 px.

Hunyuan3D-2mini runs shape-only on MPS (its texture stage needs CUDA) and needs `transformers<4.50` and `diffusers<0.33`. It is under Tencent's non-commercial license. `prep_09b` normalizes the mesh the way UniRig does (bounding box centered, longest side 1), so the fitted camera still holds after rigging. `--keep_camera` keeps that camera and the video's own mask. `compare_with_paper.py` also reports silhouette IoU, because SSIM and LPIPS penalize our frame-0 projected texture, while the paper renders the original textured asset.

The projected texture copies the near side of the animal onto the far side and smears surfaces seen edge-on. `bake_texture.py` gives the rigged mesh its own UV atlas (xatlas) and keeps frame-0 colors only where frame 0 really sees the surface, away from the silhouette and from occlusion edges. The far side is filled by mirroring across the animal's symmetry plane (found by search). A head that is turned in frame 0 does not map onto itself across the body's plane: the wolf's head is 30° off, and with one plane the far half of its face found no mirror partner and came out flat gray with no eye. So every skeleton subtree on the midline whose own plane fits it clearly better (the head, the tail) is first rotated with the rig, by linear-blend skinning so the neck follows, until its plane lines up with the body's, and the mirror is taken in that straightened pose (wolf plane error 0.021 → 0.011), with a tighter radius there (`--pose_mirror_radius 0.015`). Switching between per-part planes without straightening cut a seam across the neck ruff. Texels within `--midline 0.015` of the plane are not mirrored: they map almost onto themselves, and the nearest seen texel was the dark inner corner of the wolf's eye, a black patch between the eyes. The dog's head is not turned, so no part is straightened, and the dog keeps its first bake (equivalent to `--band_median 0 --midline 0`): the new defaults changed it only slightly, and the VLM moved both ways (+45° not-normal frames 23 → 28, −45° 13 → 5, similarity to the video 0.897 → 0.872). `--global_mirror_only` restores the single plane, and `texture_baked_source.png` maps where every texel's color came from. Thin rims take the median color of trusted texels around them (`--band_median 50`; `0` copies the single nearest texel, which turned the dark rim of the wolf's eye into a bar across its brow). A stricter `--cos_min` (0.40, 0.50) made the eye region worse, not better, and recoloring dark, obliquely seen texels (`--dark_oblique`) did not help, so it is off. Then edge-on surfaces keep the plain projection, and surfaces frame 0 never sees are filled by diffusion over the mesh. Only frame 0 is used, so the paper-metric comparison stays fair. Stage 1 renders with soft edges (`sigma` 1e-4 at 256 px blurs edges by about 4 px), which is right for optimizing and too blurry for viewing; use `--sigma 1e-6` at 512 px to look at results.

### All-around texture from multi-view generation (CUDA)

The frame-0 mesh is closed all around, but its texture comes from frame 0 alone, so the far side is mirrored or diffused. Hunyuan3D-2's own texture stage (delight, six generated views, bake) fills it properly. It needs CUDA, so it runs on a GPU server:

| Step | Where | Command |
|---|---|---|
| Install into one workspace folder (venv via uv, weights, no caches left) | server | `bash scripts/remote/setup_hy3d_paint.sh /project/<user>/animamimic_mv` |
| Paint (`shape.obj`, `frame0.png`, `frame0_mask.png` in `wolf/`) | server, one GPU | `CUDA_VISIBLE_DEVICES=0 python run_paint.py --in_dir wolf --out_dir wolf_out` |
| Move it onto our atlas | Mac | `python scripts/transfer_painted_texture.py --seq data/paper_wolf --painted outputs/paper_wolf_mv/painted.npz --out outputs/paper_wolf_mv` |
| Compare | Mac | `python tests/texture_compare.py --seq data/paper_wolf --out outputs/paper_wolf_mv/compare --variant "now=data/paper_wolf" --variant "hybrid=outputs/paper_wolf_mv/hybrid"`; `tests/texture_anim_video.py` plays a Stage 1 result with each texture |

The transfer needs no nearest-point search: shape.obj, input.glb, rigged.glb and texture_baked.npz keep the same faces in the same order, so a texel at barycentric b in face f reads the painted texture at the painted UV of the same point. The script checks this and stops if it does not hold. It writes `mv/` (generated as is), `matched/` (per-channel affine fit to frame 0, since the paint model removes lighting) and `hybrid/` (frame 0's colors where frame 0 sees the surface, matched generated colors elsewhere, feathered over 4 texels). Each folder works with `--texture_dir`. On the wolf, frame-0-view SSIM against frame 0 is 0.922 for the current bake, 0.912 for hybrid and 0.867 for the generated texture as is, and only the generated textures give the far side and the front of the face real detail. Hunyuan3D-2 is under Tencent's non-commercial license.

### Several videos at once, mostly on a GPU server

The other six project-page videos (dinosaur, cat, blue dragon, eagle, seahorse, hooded girl) ran with the GPU steps on gpu5 (one RTX 4090) and only UniRig, SAM2, AllTracker and VGGT on the Mac, because those envs exist only there and take about 4 minutes per video. Everything on the server sits in one workspace folder (`/project/<user>/animamimic_mv`), with every cache pointed inside it. The server venv is the paint venv plus `imageio`, `scikit-image`, `imageio-ffmpeg` and `lpips`; install with `uv pip install --constraint` that pins `numpy==1.26.4` and torch, or uv moves numpy to 2.x.

| Step | Where | Script | Time per video |
|---|---|---|---|
| Hunyuan3D-2 shape (full model, octree 512, 60k faces) + Hunyuan3D-Paint | server | `scripts/remote/run_shapes.sh dino cat ...` (frame 0 as `frames/<name>.png`) | ~2 min |
| Camera + fov fit (`prep_09b --fit_fov --device cuda`) | server, 3 at a time (6 GB each) | `scripts/remote/run_fits.sh paper_dino ...` | ~15 min (the Mac took over 25) |
| UniRig, reference view, frames, SAM2, AllTracker, VGGT | Mac | `scripts/run_paper_video.sh data/paper_cat data/paper_refs/ref_2.mp4` (keeps an existing fit); `scripts/remote/drive_mac_prep.sh` fetches each fit, runs it and uploads the sequence | ~4 min |
| Stage 1 (same settings as the dog and wolf) | server | `scripts/remote/run_stage1_queue.sh` runs `run_stage1.sh` on each uploaded sequence | 55 min alone, 70-80 min while Stage 2 shares the GPU (the dog took ~4 h on the Mac) |
| Bake, hybrid texture, ARAP, comparison with the paper, stretch, four-angle playback | server | `scripts/remote/run_finish_queue.sh` runs `run_finish.sh` after each Stage 1 | 3-5 min |

Results at 512 px with sharp edges, ARAP version (silhouette IoU against the SAM2 masks of the reference video; the paper's own result scored the same way):

| Sequence | IoU ours | IoU paper | Worst 1% stretch, Stage 1 → ARAP |
|---|---|---|---|
| Cat (ref 2) | 0.893 | 0.940 | 20.3% → 8.8% |
| Blue dragon (ref 3) | 0.824 | 0.871 | 14.4% → 7.2% |
| Eagle (ref 6) | 0.760 | 0.858 | 13.7% → 6.6% |
| Dinosaur (ref 1) | 0.864 | 0.927 | 17.2% → 9.1% |
| Seahorse (ref 7) | 0.835 | 0.911 | 10.0% → 5.5% |
| Hooded girl (ref 8) | 0.857 | 0.934 | 12.1% → 5.3% |

The motion follows the video in all six, and the +90° and -45° views stay whole. What is missing comes from the rig: UniRig gives the dragon's and the girl's heads one bone, so the dragon cannot open its mouth and the girl cannot wink. The eagle loses most at the feather tips. The comparison uses the frame-0 bake; the dinosaur's legs show white patches from the side there, which the hybrid texture removes (`anim_textures.png`).

### Surface stretch: ARAP after Stage 1

Linear blend skinning stretches and shears the surface where a region is blended between bones that rotate differently; on the cat the chest and shoulders sit between the spine and neck bones. Zeroing the joint translations does not reduce it (`tests/strain_ablation.py`), and more weight smoothing helps up to about 30 steps only. `scripts/arap_refine.py` keeps every frame close to the skinned result while each vertex's one-ring stays a rotated copy of its rest shape (Sorkine & Alexa 2007): bending is free, stretch and shear cost. `--lam` trades the two; at 0.01 the silhouette IoU against the video does not move on the cat (0.8905 vs 0.8904), at 0.003 it drops 0.0015 and at 0.001 0.0055. It takes 15-45 s per sequence on a CPU.

| Worst 1% of edges, mean over frames (worst frame) | Stage 1 | Stage 2 (paper) | Stage 1 + ARAP 0.01 |
|---|---|---|---|
| Wolf | 13.9% (21.5%) | 9.4% (14.8%) | 6.3% (9.7%) |
| Dog | 21.7% (27.0%) | 15.8% (20.8%) | 9.9% (13.0%) |
| Cat | 20.3% (29.2%) | 11.4% (17.6%) | 8.8% (12.3%) |

Silhouette IoU at 256 px: wolf 0.9126 / 0.9074 / 0.9099, dog 0.8259 / 0.8287 / 0.8243, cat 0.8914 / 0.8788 / 0.8912. On the cat, Stage 2 costs 0.013 IoU and ARAP nothing. `tests/surface_distortion.py` gives these numbers, `tests/strain_heatmap.py` renders the worst frame textured and as a heat map (white 0 to red 30%) with the bones whose faces stretch most, `tests/silhouette_iou.py` the fit. `run_finish.sh` applies ARAP to every sequence and shows the ARAP version in the comparison video.

`tests/paper_overview_video.py` stacks the `compare_with_paper` videos of several sequences into one, a row each, labeled with the silhouette IoU of ours and of the paper. `scripts/finish_paper_video.sh` is the Mac version of the last step.

## Animate an existing rigged asset from a generated video

This is the paper's use case without UniRig or mesh reconstruction: an artist-rigged GLB keeps its own skeleton and skin weights, a video model animates a render of it, and Stage 1 transfers that motion back onto the original rig. Example: `data/hf_cow` ("Cow NPC" by nataliekirk, CC BY 4.0, from the Hugging Face mirror `Linzhan/Objaverse-XL-Rigged-Animated`; see `data/hf_cow/CREDIT.md`).

| Step | Env | Command | Output |
|---|---|---|---|
| Put the asset in a sequence folder | | copy the GLB to `data/hf_cow/rigged.glb` | |
| Render the still | animamimic | `python scripts/prep_01_render_view.py --seq data/hf_cow --azim -30 --dist 1.8 --elev 12` | `reference.png`, `camera.json` |
| Generate the video | `third_party/venvs/ltx` | `ltx-2-mlx generate --model models/ltx-2.3-mlx-q4 --gemma models/gemma-3-12b-it-4bit --distilled -H 512 -W 512 -f 97 --frame-rate 24 --image ../data/hf_cow/reference.png -p "<motion>. The cow stays in the same spot ... Static camera ..." -o ../data/hf_cow/candidates/graze.mp4` (from `third_party/`) | `.mp4` |
| Import, masks, tracks, depth | as for any sequence | `prep_02 --video data/hf_cow/candidates/graze.mp4`, `prep_04`, `prep_05`, `prep_06` | `frames/`, `masks/`, `tracks.npz`, `depth.npz` |
| Stage 1 | animamimic | `python scripts/run_animamimic.py --seq data/hf_cow --out outputs/hf_cow --device mps --backend raster --image_size 384 --sigma 1e-5 --w_rigid 10 --iters 3000 --batch_frames 4 --depth_norm per_frame --trans_max 0.01 --w_reg 1000 --skin_smooth 0` | `stage1.npz` |
| Write the motion into the GLB | animamimic | `python scripts/export_animated_glb.py --seq data/hf_cow --stage1 outputs/hf_cow/stage1.npz --duration 4.0 --name graze --out outputs/hf_cow/cow_animated.glb` | `cow_animated.glb` |
| Look at it | animamimic | `python tests/frame_qa_render.py --seq data/hf_cow --result outputs/hf_cow/stage1.npz --out outputs/hf_cow/frame_qa --texture rig --device cpu`, then `python tests/rig_animation_video.py --qa outputs/hf_cow/frame_qa --out outputs/hf_cow/cow_vs_video` | `.mp4`, `.png` |

After the video exists, `zsh scripts/run_hf_asset.sh data/hf_bear data/hf_bear/candidates/sniff.mp4 sniff` runs every step from the import to the side-by-side video. Four assets are done this way: `hf_cow`, `hf_bear`, `hf_elephant`, `hf_moose` (input-view IoU 0.875, 0.912, 0.886, 0.815). Prompts that keep the animal in place and move only its head or trunk work. Prompts that turn or shake the head usually make LTX turn the whole animal, and LTX reshapes the moose's low-poly antlers while they move. Render the still close to a side view: from a three-quarter view LTX rotated the moose into profile.

Keep `--skin_smooth 0` for hand-made weights: the smoothing exists for UniRig's weights, which jump between joints across single edges. `--duration` is the time from the first to the last fitted frame (32 frames picked from 97 at 24 fps: 96 / 24 = 4.0 s). The exporter keys every joint's local translation, rotation and scale at each frame and leaves the mesh, texture, skin and skeleton unchanged; it replaces the asset's own animations unless `--keep_original` is given. It plays the result back with our glTF reader and prints the largest vertex error against `stage1.npz` (2e-7 for the cow; Blender's importer reproduces the same vertices to 4e-7). Sketchfab exports store some static ancestor nodes as raw matrices, which `gltf_rig.py` now reads. Stage 2 cannot be exported this way, because the simulation moves vertices that no skeleton pose reproduces.

## Check every frame with independent image models

```bash
python tests/frame_qa_render.py --seq data/paper_dog --result outputs/paper_dog/stage1.npz --out outputs/paper_dog/frame_qa
third_party/venvs/vlm/bin/python tests/frame_qa_vlm.py --dir outputs/paper_dog/frame_qa --animal dog
python tests/frame_qa_report.py --dir outputs/paper_dog/frame_qa
```

Every frame is rendered from the input view, +/-45 degrees (the paper's novel views) and +90 degrees. DINOv2 gives each frame its similarity to the previous frame (glitches) and to the undeformed mesh from the same view (drift). Qwen3-VL-8B (4-bit MLX) answers, for each image, whether the animal looks normal, with a 1-10 score and the problem it sees. The undeformed mesh and the reference video are scored the same way as baselines. `report.json` lists the flagged frames and `flagged.png` shows them. `python scripts/compare_versions_video.py --paper data/paper_refs/ours_4.mp4 --version "old=outputs/paper_dog/frame_qa" --version "new=outputs/paper_dog_sharp_rigid/frame_qa" --out versions` puts several versions side by side (input view, +45°, side) with each frame's VLM score on it. The VLM judges a frame against an ideal animal, so a frame cannot score above the undeformed mesh it was made from; compare each frame with the rest-mesh score of the same view. At 512 px full body the VLM judges the whole animal and misses face problems (it scored the wolf's flat gray half-face and the fixed face the same at side 90°); `tests/head_qa_render.py` renders head close-ups of every frame in the same layout, to score with `--animal "wolf's head"`. The VLM also prefers a plain, featureless half-face over one with small texture flaws, so read its face scores next to the images.

`tests/frame_sim_vlm.py --dir <frame_qa dir> --animal wolf` compares every frame with the original video frame: the VLM sees both and answers yes/no questions (input view: same pose? same face, fur colors and markings? novel views: the same animal from that angle?), and the score is its probability of answering Yes (VQAScore). Asking for a 1–10 score (`--mode score`) gave nearly every frame of every version 9, and a forced choice between two renders (`--versus`) picked the second image every time, so only the probability separates versions. `--extract_paper ours_N.mp4` scores the paper's input-view result the same way, `--rest` the undeformed mesh, and `tests/frame_sim_report.py` tabulates them. Cross-view scores favor renders that look like the video image itself (a face turned toward that camera), so use them to compare versions with the same geometry, not as a measure of 3D correctness. The `vlm` env is `uv venv -p 3.11 third_party/venvs/vlm && uv pip install --python third_party/venvs/vlm/bin/python mlx-vlm requests`.

## Tests

```bash
python -m pytest tests/test_core.py -q
python scripts/validate_fox.py --device cuda --image_size 256 --iters 3000
```

`tests/paper_metrics.py` computes the paper's Table 1 SSIM and LPIPS of the input-view render against the video. `tests/depth_target_quality.py` compares depth targets with the truth on the fox.

`validate_fox.py` has a known answer: a rigged fox is animated and rendered, and Stage 1 must recover the 3D motion from one view. It reports 3D vertex error and novel-view scores at +/-45 degrees.

## Differences from the paper

| Item | Paper | Here | Reason |
|---|---|---|---|
| Physics engine | NVIDIA Warp | Batched PyTorch, float64 | Same equations (Eq. 15-21). Checked against finite differences. |
| Tet meshing | TetWild | fTetWild through `pytetwild` | Successor by the same group, installable with pip. |
| Video model | Kling (paid) | Any image-to-video `.mp4` | Generate the video separately. |
| Skin weights | UniRig output | Optional Laplacian smoothing, `--skin_smooth N` (off by default) | UniRig's weights can switch joints across one edge, which tore the horse's head and neck. |
| Edge-length term | None | Optional, `--w_rigid` (off by default) | Leaf joints rotated freely and creased the dog's face. |
| Stretch removal after Stage 1 | Stage 2 physics | Also `scripts/arap_refine.py` (ARAP projection per frame, `--lam 0.01`) | Linear blend skinning stretches regions blended between bones; ARAP removes more of it than Stage 2, in seconds. |

## Values the paper does not publish

These are placeholders in `animamimic/stage1.py` and `animamimic/stage2.py`. Results depend on them.

- Loss weights: lambda for RGB, mask, track, depth, smooth, reg.
- Iteration counts for both stages, and frames rendered per iteration.
- Thresholds theta-hat for rotation and translation in Eq. 9. With the defaults (`trans_max` 0.05, `w_reg` 1), Stage 1 moved the horse almost entirely by bone translation and tore the mesh. The horse results use `--trans_max 0.01 --w_reg 1000 --skin_smooth 30`. The defaults stay unchanged until the Fox validation is rerun with these values.
- The depth normalization operator N(.) in Eq. 8. `first_frame` and `per_frame` are both implemented. On the fox, per-point VGGT depth was twice as far from the truth as assuming no depth motion at all, even with an ideal per-frame scale and shift, so N(.) is not the main problem. `--depth_mode mean` (not in the paper) matches only each frame's mean depth: it keeps the body from sliding toward the camera and leaves the shape to the image terms. The default stays `points`, as in the paper.
- Video length and resolution, renderer lighting and blur settings.
- Number of AllTracker points.
- Initial Young's modulus, Poisson's ratio, density, time step and substeps. The time step must follow the clip: `--fps` is the frame rate of the loaded frames, (frames - 1) / clip seconds. The paper videos are 121 frames at 24 fps sampled to 32, so 6.2; the default 24 plays the motion 3.9 times too fast and exaggerates inertia.
- Coarse-to-fine cluster schedule for Stage 2.

Stage 2 is driven as the paper describes it (Sec. 3.2.2): each joint's transform is the boundary condition of the tet nearest the joint, implemented by pinning that tet's four nodes. Newton starts every time step from the skinned pose of all tet nodes and stops when the force residual has dropped by `newton_rel_tol` (1e-4). Starting from the inertial prediction with only the pins moved stretched the tets around the pins, and Newton then ran into its 30-iteration cap on every step; the old stopping test alone scales with 1/h^2 and asks for far more precision at the real frame rate. Stage 2 renders with Stage 1's `--sigma` and normalizes depth with Stage 1's `--depth_norm`. One iteration takes about 6.5 minutes on the wolf (9,829 nodes, 38,035 tets) on the Mac CPU, and more than 6 PyTorch threads does not help. On a CUDA device the physics runs on the GPU in float64; on gpu5's RTX 4090 the cat (8,128 nodes, 31,774 tets) took about 21 minutes per iteration and 6.5 hours for `--stage2_iters 10 10`, with one CPU core busy the whole time and Stage 1 jobs sharing the GPU. Consumer GPUs are slow in float64 and these solves are small, so the CPU is likely the faster choice there too (not measured on the server). The surface follows the tets through barycentric weights in the nearest tet, as in the paper, without clamping: about 70% of the surface vertices lie slightly outside the fTetWild mesh, and clamped weights moved the rest surface by up to half an edge, which added about 2% of stretch everywhere and hid what the simulation does. With the paper's learning rate the moduli move by at most 2% in 20 iterations (`--stage2_iters 10 10`), so Stage 2's effect is the simulation itself. `--stage2_E stage2.npz` simulates with saved moduli without optimizing, and `stage2.npz` keeps the tet node paths (`tet_x`). `python tests/surface_distortion.py --seq <seq> --result stage1.npz stage2.npz` measures what the paper claims for Stage 2, less surface distortion: on the wolf the worst 1% of edge stretch drops from 13.9% to 9.4% and the mean from 2.0% to 1.6%; on the dog from 21.7% to 15.8% and 2.7% to 2.4%.

The paper states these directly, so they are used as written: Adam with learning rate 1e-3, 6D rotations, UniRig, SAM2, AllTracker, VGGT, Fixed Corotated elasticity, backward Euler with Newton and line search, no external forces, optimizing log Young's modulus.
