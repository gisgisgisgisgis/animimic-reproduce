# lida/：AnimaMimic（AniMimic）重現，論文影片路線

> 這個資料夾是 Lida 的獨立實作，和上層的 `src/animimic` 是兩套程式，互不相依，也不會改動上層的任何檔案。**所有指令都在 `lida/` 裡執行。** 兩套實作各自怎麼做，見 [COMPARISON.md](COMPARISON.md)。

獨立重寫 *AniMimic: Imitating 3D Animation from Video Priors*（CVPR 2026，arXiv 2512.14133，arXiv 版標題是 AnimaMimic）。作者沒有公開程式碼，這裡的程式全部依照論文重寫。

給一段動物影片，做出一隻照著影片動的 3D 動物：

```text
影片（論文網頁的參考影片，或自己用影片模型生成的影片）
  → 第 0 幀建模（Hunyuan3D-2）→ 相機擬合 → 自動骨架（UniRig）
  → SAM2 輪廓 + AllTracker 追蹤點 + VGGT 深度
  → Stage 1：最佳化每一格每個關節的角度
  → ARAP 去除表面拉扯（論文沒有，我們加的）→（選用）Stage 2：FEM 物理
  → 和論文結果並排的影片、分數、四個角度的播放、骨架動畫 GLB
```

不需要訓練模型。所有步驟都是「現成模型 + 最佳化」。

| 動物 | 輪廓 IoU 我們 | 論文 | 表面最糟 1% 拉扯：Stage 1 → ARAP |
| --- | --- | --- | --- |
| 貓 | 0.893 | 0.940 | 20.3% → 8.8% |
| 藍龍 | 0.824 | 0.871 | 14.4% → 7.2% |
| 老鷹 | 0.760 | 0.858 | 13.7% → 6.6% |

論文網頁上的 8 段影片全部跑完了，8 段的數字在[第 4 節](#4-結果)。結果影片和圖片含有論文影片的畫面，所以不放在這個公開的 repo 裡；需要的話請找 Lida。

## 文件導覽

| 文件 | 內容 |
| --- | --- |
| **README.md**（本頁） | 怎麼安裝、怎麼跑 |
| [COMPARISON.md](COMPARISON.md) | 和上層 `src/animimic` 的對照：兩邊各自怎麼做、可以互相借用什麼 |
| [docs/UNDERSTANDING.md](docs/UNDERSTANDING.md) | 怎麼理解這個專案：論文在做什麼、每一步為什麼這樣做、數字怎麼看 |
| [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) | 英文技術筆記：每個腳本和參數的細節 |
| `docs/3d_basics_tutorial.html` | 互動教材：mesh、UV、骨架、LBS、ARAP、可微分渲染（下載後用瀏覽器開） |

## 腳本一覽

平台欄：🍎 只能在 Mac（Apple Silicon）上跑；🐧 只能在 Linux + NVIDIA GPU 上跑；兩者都寫的表示程式本身不分平台。

| 腳本 | 平台 | 用途 |
| --- | --- | --- |
| `scripts/setup/clone_third_party.sh [名稱 ...]` | 🍎 🐧 | 把 SAM2、VGGT、AllTracker、UniRig、Hunyuan3D-2、ltx-2-mlx clone 到 `third_party/`，固定在我們跑過的版本 |
| `scripts/setup/install_envs_mac.sh` | 🍎 | 建立 SAM2 + VGGT + AllTracker 用的 `prep` 環境，以及 UniRig 環境 |
| `scripts/setup/install_ltx_mac.sh` | 🍎 | 選用：LTX-2.3 影片生成（MLX），只有自己生成影片時才需要 |
| `scripts/setup/download_paper_refs.sh` | 🍎 🐧 | 從論文網頁下載 8 段參考影片和論文自己的結果到 `data/paper_refs/` |
| `scripts/remote/setup_hy3d_paint.sh <工作區>` | 🐧 | 在一個資料夾裡建 venv、裝 Hunyuan3D-2 和上色模型，不需要 sudo |
| `scripts/run_paper_video.sh <序列> <影片>` | 🍎 | 相機擬合 → UniRig → 切格 → SAM2 → AllTracker → VGGT（用 Mac 版 UniRig 和 `caffeinate`） |
| `scripts/run_animamimic.py` | 🍎 🐧 | Stage 1（以及選用的 Stage 2），`--device mps` 或 `cuda` |
| `scripts/arap_refine.py` | 🍎 🐧 | Stage 1 之後去除表面拉扯（CPU） |
| `scripts/finish_paper_video.sh <序列> <N>` | 🍎 | 烘焙貼圖、混合貼圖、ARAP、和論文比較、拉扯、四角度播放 |
| `scripts/remote/run_*.sh` | 🐧 | 伺服器版：建模、相機擬合、Stage 1、收尾（含 ARAP）、Stage 2，可排隊跑多隻 |
| `scripts/remote/drive_mac_prep.sh`、`fetch_paper_result.sh` | 🍎 | 在 Mac 上做前處理、上傳到伺服器、把結果下載回來 |
| `scripts/run_hf_asset.sh <序列> <影片> <名稱>` | 🍎 | 現成骨架模型：從匯入影片一路做到骨架動畫 GLB 和對照影片 |
| `tests/paper_overview_video.py` | 🍎 🐧 | 把多隻動物的對照影片疊成一支總覽 |

`.sh` 腳本中開頭是 `#!/bin/zsh` 的（`run_paper_video.sh`、`finish_paper_video.sh`、`run_hf_asset.sh`、`remote/drive_mac_prep.sh`、`remote/fetch_paper_result.sh`）是給 Mac 用的；在 Linux 上要先裝 zsh。

## 哪些在哪裡跑過、別人能不能直接跑

✅ 我們實際跑過；— 沒跑過；✗ 不能跑。

| 步驟 | Mac（Apple Silicon） | Linux + NVIDIA GPU |
| --- | --- | --- |
| 第 0 幀建模（Hunyuan3D-2） | ✅ 2mini，MPS | ✅ 完整版，`run_shapes.sh` |
| 上色（Hunyuan3D-Paint） | ✗ 只支援 CUDA | ✅ |
| 相機擬合（`prep_09b`） | ✅ 但很慢（25 分鐘以上） | ✅ `--device cuda`，約 15 分鐘 |
| UniRig | ✅ `prep_00_unirig_mac.sh`（CPU + `mac_shims/`） | — `prep_00_unirig.sh` 寫好了，沒跑過 |
| 切格、SAM2、AllTracker、VGGT | ✅ MPS | — 程式不分平台，但我們沒在 Linux 上跑過 |
| Stage 1 | ✅ MPS，約 2–4 小時 | ✅ CUDA，約 1 小時 |
| ARAP、烘焙貼圖、混合貼圖、和論文比較、拉扯 | ✅ `finish_paper_video.sh` | ✅ `run_finish.sh` |
| 四面體（`prep_07`）、Stage 2 | ✅ 物理在 CPU | ✅ 四面體在 Mac 做；物理在 GPU 上反而慢 |
| LTX-2.3 影片生成、Qwen3-VL 逐格檢查 | ✅ MLX，只有 Mac | ✗ |
| `test_core.py`、`validate_fox.py` | ✅ | — |

**別人能不能直接跑？**

- **Mac**：可以，整條路線都在 Mac 上跑過。不過環境要自己建（第 3 節 A），第一次大約要下載 10–20 GB 的套件和模型權重。
- **Linux + NVIDIA GPU**：Stage 1、收尾、建模、上色、相機擬合都在伺服器上跑過。UniRig、SAM2、AllTracker、VGGT 這幾個前處理在 Linux 上沒跑過；要嘛在 Mac 上做完再上傳（第 6 節的做法），要嘛自己在 Linux 上裝這些工具。
- **資料不在 Git 裡**：論文影片用 `download_paper_refs.sh` 下載，其他（模型、骨架、追蹤點、深度、Stage 1 結果）都要照流程重新產生。

## 1. 系統與版本

這些是我們實際跑過的環境。每個環境完整的套件清單在 [env/](env/)（`uv pip freeze` 輸出）。

| 機器 | 用途 | 版本 |
| --- | --- | --- |
| Mac（Apple M5 Pro，48 GB） | UniRig、SAM2、AllTracker、VGGT；也能跑全部步驟（較慢） | Python 3.11.15；主環境 torch 2.5.1、numpy 1.26.4、PyTorch3D 0.7.9（選用） |
| Linux GPU 伺服器（用一張 RTX 4090） | Hunyuan3D-2 建模和上色、相機擬合、Stage 1、收尾、Stage 2 | Python 3.12.3、driver 580、torch 2.5.1+cu124、numpy 1.26.4、diffusers 0.31.0、transformers 4.46.3 |

| 環境 | 位置 | 套件清單 |
| --- | --- | --- |
| 🍎 主環境（Stage 1/2、渲染、比較） | `.venv` | [env/mac-animamimic.txt](env/mac-animamimic.txt) |
| 🍎 SAM2、VGGT、AllTracker | `third_party/venvs/prep` | [env/mac-prep.txt](env/mac-prep.txt) |
| 🍎 UniRig | `third_party/venvs/unirig` | [env/mac-unirig.txt](env/mac-unirig.txt) |
| 🍎 Hunyuan3D-2（只做形狀） | `third_party/venvs/hunyuan3d` | [env/mac-hunyuan3d.txt](env/mac-hunyuan3d.txt) |
| 🍎 LTX-2.3（選用，MLX） | `third_party/venvs/ltx` | [env/mac-ltx.txt](env/mac-ltx.txt) |
| 🐧 伺服器（上色 + 主程式） | `<工作區>/venv` | [env/server-cuda.txt](env/server-cuda.txt) |

程式的裝置順序是 cuda → mps → cpu。在 CUDA 伺服器上實際跑過的是 `--backend raster`（我們自己用 PyTorch 寫的軟光柵），PyTorch3D 的 CUDA 路線（`setup_cuda.sh`）寫好了但還沒完整跑過。Stage 2 的物理要用 float64：Mac 的 MPS 不支援，會自動改在 CPU 上算。

## 2. 下載專案

```bash
git clone https://github.com/gisgisgisgisgis/animimic-reproduce.git
cd animimic-reproduce
git checkout lida/animamimic-reproduce   # 合併進 main 之前
cd lida
```

後面的相對路徑都以 `lida/` 為準。`data/`、`outputs/`、`third_party/` 和各個環境都不在 Git 裡，要自己建立或下載。

## 3. 建立環境

### A. 🍎 Mac（Apple Silicon）

這一節的環境清單（`env/mac-*.txt`）是在 Mac 上匯出的，裡面的 torch 等套件是 Mac 版，不能直接拿到 Linux 上裝。

需要 [uv](https://docs.astral.sh/uv/)。

```bash
# 主環境
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r env/mac-animamimic.txt
```

`env/mac-animamimic.txt` 裡的 PyTorch3D 會從原始碼編譯，比較久。它只有 `--backend pytorch3d` 會用到；只用 `--backend raster` 的話，可以先把那一行拿掉。

```bash
# 上游工具和它們的環境
bash scripts/setup/clone_third_party.sh
bash scripts/setup/install_envs_mac.sh          # prep（SAM2、VGGT、AllTracker）和 UniRig
uv venv --python 3.11 third_party/venvs/hunyuan3d
uv pip install --python third_party/venvs/hunyuan3d/bin/python -r env/mac-hunyuan3d.txt
```

`install_envs_mac.sh` 只裝了基本套件。缺什麼的話，對照 `env/mac-unirig.txt`、`env/mac-prep.txt` 補裝。UniRig 在 Mac 上的 `spconv`、`flash_attn` 沒有 Mac 版，`scripts/prep_00_unirig_mac.sh` 會自動換成 `scripts/mac_shims/` 裡的純 PyTorch 版本，並且改用 CPU（`torch_cluster` 在 MPS 上會報錯）。

長時間的步驟記得防止 Mac 休眠：在指令前加 `caffeinate -is`。

### B. 🐧 Linux GPU 伺服器（不需要 sudo）

所有東西放在同一個工作區資料夾，例如 `/project/<user>/animamimic_mv`，快取也指到裡面，不寫到家目錄。

```bash
# 在伺服器上
bash setup_hy3d_paint.sh /project/<user>/animamimic_mv
```

這會用 uv 建立 `venv`（torch 2.5.1+cu124）、clone Hunyuan3D-2、編譯上色用的 rasterizer、下載上色模型。`CUDA_HOME` 預設是 `/usr/local/cuda-12.4`，伺服器上的版本不同時要先設定；`TORCH_CUDA_ARCH_LIST` 預設 8.9（RTX 4090）。

主程式需要的套件，再補裝到同一個 venv。一定要加 `--constraint`，不然 uv 會把 numpy 升到 2.x：

```bash
cat > constraints.txt <<'EOF'
numpy==1.26.4
torch==2.5.1+cu124
torchvision==0.20.1+cu124
EOF
uv pip install --python venv/bin/python --constraint constraints.txt \
  imageio imageio-ffmpeg "scikit-image==0.24.0" "tifffile==2024.9.20" lpips
```

完整清單在 `env/server-cuda.txt`。另外 `setup_cuda.sh` 是用 conda 建三個環境的寫法（含 PyTorch3D），寫好但還沒完整跑過。

## 4. 結果

512×512、32 格、對同一段原始影片計分，用的是 ARAP 之後的版本。拉扯是每一格最糟 1% 的邊被拉長的比例，對 32 格平均。

| 動物 | 影片 | IoU 我們 | IoU 論文 | SSIM 我們 | SSIM 論文 | LPIPS 我們 | LPIPS 論文 | 拉扯 Stage 1 → ARAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 貓 | 2 | 0.893 | 0.940 | 0.893 | 0.902 | 0.088 | 0.059 | 20.3% → 8.8% |
| 藍龍 | 3 | 0.824 | 0.871 | 0.931 | 0.933 | 0.065 | 0.058 | 14.4% → 7.2% |
| 老鷹 | 6 | 0.760 | 0.858 | 0.911 | 0.922 | 0.098 | 0.051 | 13.7% → 6.6% |
| 恐龍 | 1 | 0.864 | 0.927 | 0.922 | 0.948 | 0.081 | 0.053 | 17.2% → 9.1% |
| 海馬 | 7 | 0.835 | 0.911 | 0.936 | 0.949 | 0.073 | 0.046 | 10.0% → 5.5% |
| 少女 | 8 | 0.857 | 0.934 | 0.929 | 0.955 | 0.074 | 0.039 | 12.1% → 5.3% |
| 白狼 | 5 | 0.913 | 0.937 | 0.909 | 0.931 | 0.080 | 0.057 | 13.9% → 6.3% |
| 小狗 | 4 | 0.825 | 0.922 | 0.920 | 0.942 | 0.083 | 0.052 | 21.7% → 9.9% |

IoU、SSIM 越高越好，LPIPS、拉扯越低越好。我們的貼圖是從第 0 幀重建的，論文用原本就有貼圖的模型，所以 SSIM、LPIPS 先天吃虧；輪廓 IoU 不看貼圖，比較公平。論文的 Stage 2 在白狼、小狗、貓上把拉扯降到 9.4%、15.8%、11.4%，ARAP 降得更多，而且只要幾十秒。

現成綁好骨架的模型加上 LTX-2.3 生成的影片（第 7 節）：乳牛 0.875、熊 0.912、大象 0.886、麋鹿 0.815（輸入視角的輪廓 IoU）。有標準答案的 Fox 驗證：頂點平均誤差 0.70%（完全不動是 3.87%）。

## 5. 🍎 跑一段論文網頁的影片（全部在 Mac 上）

下面的指令是 Mac 版：UniRig 用 `prep_00_unirig_mac.sh`，Stage 1 用 `--device mps`，`caffeinate -is` 防止休眠。在 Linux 上，GPU 的步驟改用第 6 節的伺服器腳本；前處理（UniRig、SAM2、AllTracker、VGGT）我們沒在 Linux 上跑過。

以貓（第 2 段）為例。每一步的輸入、輸出都在 `data/paper_cat/`。

```bash
# 1. 下載論文網頁的影片（ref_N.mp4 是輸入，ours_N.mp4 是論文的結果）
bash scripts/setup/download_paper_refs.sh

# 2. 從第 0 幀建模（Mac 預設用 Hunyuan3D-2mini；我們最後用的是完整版，見第 6 節）
third_party/venvs/hunyuan3d/bin/python scripts/prep_09a_frame0_to_3d.py \
  --video data/paper_refs/ref_2.mp4 --seq data/paper_cat

# 3. 相機擬合、UniRig、切格、SAM2、AllTracker、VGGT
caffeinate -is zsh scripts/run_paper_video.sh data/paper_cat data/paper_refs/ref_2.mp4

# 4. Stage 1（Mac 上約 2–4 小時；CUDA 上把 mps 換成 cuda，約 1 小時）
caffeinate -is .venv/bin/python scripts/run_animamimic.py --seq data/paper_cat \
  --out outputs/paper_cat_sharp_rigid --device mps --backend raster --image_size 384 --sigma 1e-5 \
  --w_rigid 10 --iters 3000 --batch_frames 4 --depth_norm per_frame \
  --trans_max 0.01 --w_reg 1000 --skin_smooth 30

# 5. 去除表面拉扯
.venv/bin/python scripts/arap_refine.py --seq data/paper_cat \
  --result outputs/paper_cat_sharp_rigid/stage1.npz \
  --out outputs/paper_cat_sharp_rigid/stage1_arap.npz --lam 0.01

# 6. 烘焙貼圖、和論文比較、四角度播放
zsh scripts/finish_paper_video.sh data/paper_cat 2
```

Stage 1 的參數為什麼這樣設，見 [docs/UNDERSTANDING.md](docs/UNDERSTANDING.md#stage-1讓骨架模仿影片)。`--device mps` 時 Stage 2 的物理會自動改在 CPU 上跑。

和論文比較、拉扯數字、熱圖可以單獨跑：

```bash
.venv/bin/python tests/compare_with_paper.py --seq data/paper_cat --paper data/paper_refs/ours_2.mp4 \
  --result outputs/paper_cat_sharp_rigid/stage1_arap.npz outputs/paper_cat_sharp_rigid/stage1.npz \
  --texture baked --size 512 --sigma 1e-6 --out outputs/paper_cat_sharp_rigid/compare_with_paper_baked_512
.venv/bin/python tests/surface_distortion.py --seq data/paper_cat \
  --result outputs/paper_cat_sharp_rigid/stage1.npz outputs/paper_cat_sharp_rigid/stage1_arap.npz
.venv/bin/python tests/strain_heatmap.py --seq data/paper_cat \
  --result outputs/paper_cat_sharp_rigid/stage1.npz outputs/paper_cat_sharp_rigid/stage1_arap.npz \
  --label "Stage 1" "ARAP" --out outputs/paper_cat_sharp_rigid/strain
```

### 選用：Stage 2（FEM）

```bash
.venv/bin/python scripts/prep_07_tetmesh.py --seq data/paper_cat
caffeinate -is .venv/bin/python scripts/run_animamimic.py --seq data/paper_cat \
  --out outputs/paper_cat_sharp_rigid --device mps --backend raster --image_size 384 --sigma 1e-5 \
  --batch_frames 4 --depth_norm per_frame --skin_smooth 30 \
  --stage1_from outputs/paper_cat_sharp_rigid/stage1.npz --stage2 --stage2_iters 10 10 --fps 6.2
```

`--fps` 是取出來那 32 格的幀率：論文影片是 121 格、24 fps，所以 (32 − 1) / (120 / 24) = 6.2；第 6、7 段是 30 fps，要用 7.75。Stage 2 很慢：白狼在 Mac CPU 上每次迭代約 6.5 分鐘，20 次約 2 小時。

### 多隻疊成一支總覽

```bash
third_party/venvs/prep/bin/python tests/paper_overview_video.py --out outputs/overview \
  --item "貓=outputs/paper_cat_sharp_rigid/compare_with_paper_baked_512" \
  --item "藍龍=outputs/paper_dragon_sharp_rigid/compare_with_paper_baked_512"
```

## 6. 🍎 + 🐧 Mac 和 GPU 伺服器分工（我們實際的做法）

GPU 吃重的步驟丟到伺服器，只有 UniRig、SAM2、AllTracker、VGGT 留在 Mac（這幾個環境只在 Mac 上建好，每段影片約 4 分鐘）。

伺服器上的工作區長這樣，`scripts/remote/*.sh` 也放在工作區最上層：

```text
<工作區>/
  venv/  hf/  Hunyuan3D-2/      ← setup_hy3d_paint.sh 建的
  code/                         ← 上傳 animamimic/、scripts/、tests/
  frames/<名稱>.png             ← 每段影片的第 0 幀
  seqs/paper_<名稱>/            ← 每段影片的序列（Mac 做完前處理後上傳）
  stage1/  finish/  stage2/     ← 結果
```

Mac 端的腳本用兩個環境變數找伺服器（ssh 要先能免密碼登入）：

```bash
export ANIMAMIMIC_HOST=<user>@<server>
export ANIMAMIMIC_REMOTE_ROOT=/project/<user>/animamimic_mv
```

| 步驟 | 在哪 | 腳本 | 每段影片的時間 |
| --- | --- | --- | --- |
| Hunyuan3D-2 完整版建模（octree 512、6 萬面）+ 上色 | 伺服器 | `bash run_shapes.sh dino cat ...` | 約 2 分鐘 |
| 相機 + 視野角擬合 | 伺服器，一次最多 3 個（各約 6 GB） | `bash run_fits.sh paper_dino paper_cat ...` | 約 15 分鐘 |
| UniRig、切格、SAM2、AllTracker、VGGT，上傳 | Mac | `zsh scripts/remote/drive_mac_prep.sh` | 約 4 分鐘 |
| Stage 1 | 伺服器 | `bash run_stage1_queue.sh paper_cat ...` | 單獨跑 55 分鐘 |
| 收尾：烘焙、混合貼圖、ARAP、比較、拉扯、播放 | 伺服器 | `bash run_finish_queue.sh paper_cat:2 ...` | 3–5 分鐘 |
| 下載回 Mac | Mac | `zsh scripts/remote/fetch_paper_result.sh paper_cat` | 幾秒 |
| Stage 2（選用） | 伺服器 | `bash run_stage2.sh paper_cat 6.2` | 約 6.5 小時（見下方） |

所有伺服器腳本都只用一張 GPU（`CUDA_VISIBLE_DEVICES=0`），也把 `TMPDIR`、`HF_HOME`、`TORCH_HOME` 等快取指到工作區。共用伺服器上請依實驗室規則選卡。

Stage 2 在 RTX 4090 上反而比 Mac CPU 慢（貓每次迭代約 21 分鐘）：物理用 float64，遊戲卡的 float64 很慢，方程式又小。之後在伺服器上跑 Stage 2，物理大概也該放 CPU（還沒在伺服器上實測）。

## 7. 🍎 讓現成綁好骨架的模型動起來

這是論文真正的使用情境：模型保留藝術家綁好的骨架和權重，用影片模型生成它怎麼動，再用 Stage 1 把動作抄回它自己的骨架。

```bash
# 1. 把 GLB 放進序列資料夾：data/hf_cow/rigged.glb
# 2. 拍一張截圖
.venv/bin/python scripts/prep_01_render_view.py --seq data/hf_cow --azim -30 --dist 1.8 --elev 12
# 3. 用 LTX-2.3 生成影片（只能在 Mac 上：MLX；需要 install_ltx_mac.sh，在 third_party/ 裡執行）
cd third_party && venvs/ltx/bin/ltx-2-mlx generate --model models/ltx-2.3-mlx-q4 \
  --gemma models/gemma-3-12b-it-4bit --distilled -H 512 -W 512 -f 97 --frame-rate 24 \
  --image ../data/hf_cow/reference.png \
  -p "The cow lowers its head and grazes. The cow stays in the same spot. Static camera." \
  -o ../data/hf_cow/candidates/graze.mp4 && cd ..
# 4. 從匯入影片一路做到動畫 GLB 和對照影片
zsh scripts/run_hf_asset.sh data/hf_cow data/hf_cow/candidates/graze.mp4 graze
```

模型可以從 Hugging Face 的 `Linzhan/Objaverse-XL-Rigged-Animated` 找，用的時候要照授權標註作者。手綁的權重不需要平滑，`run_hf_asset.sh` 用 `--skin_smooth 0`。影片提示詞要讓動物「待在原地、只動頭」，叫它轉頭或甩頭時，LTX 常會讓整隻轉過去。

## 8. 輸出檔案

| 檔案 | 內容 |
| --- | --- |
| `stage1.npz` | 每一格每個關節的旋轉（6D）和位移、每一格的頂點位置 |
| `stage1_arap.npz` | 同上，頂點換成 ARAP 之後的位置 |
| `stage2.npz` | Stage 2 的頂點、四面體節點軌跡和楊氏模數 |
| `stage1.json` / `stage2.json` | 每 100 步的各項 loss |
| `compare_with_paper_baked_512.{mp4,png,json}` | 五欄對照影片和分數（SSIM、LPIPS、輪廓 IoU） |
| `surface_distortion.json` | 每一格的邊長拉扯（平均、最糟 1%）和抖動 |
| `strain.png` | 最糟那一格的拉扯熱圖（白 0%，紅 30%） |
| `anim_textures.{mp4,png}` | 四個角度播放，兩種貼圖並排 |
| `*_animated.glb` | 骨架動畫寫回原本的 GLB（Blender 可直接開） |

## 9. 測試與驗證

```bash
.venv/bin/python -m pytest tests/test_core.py -q
.venv/bin/python scripts/validate_fox.py --device cuda --image_size 256 --iters 3000
```

`test_core.py` 有 13 項：FK/LBS、相機、FEM 的梯度和 Hessian、adjoint 對有限差分、從運動回推已知的剛性。`validate_fox.py` 有標準答案：用 Fox 自己的動畫渲染影片，只看影片還原 3D 動作，我們得到頂點平均誤差 0.70%（完全不動是 3.87%）。

## 10. 常見問題

| 現象 | 處理方式 |
| --- | --- |
| 裝套件後 numpy 變成 2.x | 用 `uv pip install --constraint` 固定 numpy 1.26.4 和 torch |
| 骨架和模型錯位，腿被撕成鋸齒 | `prep_00_unirig_mac.sh` 會先用 `prep_00a_bake_mesh.py` 把節點變換烘焙進模型；`prep_01` 在頂點到骨頭的距離中位數超過 0.15 時會直接停下 |
| 動作時模型被撕開、骨頭飄走 | 用 `--trans_max 0.01 --w_reg 1000 --skin_smooth 30` |
| 臉被扯皺、耳朵捲起來 | 用 `--w_rigid 10` |
| 動起來表面被拉長、關節塌陷 | Stage 1 之後跑 `arap_refine.py --lam 0.01` |
| 轉到側面，背面顏色是錯的 | `bake_texture.py`（對稱鏡射），或在伺服器跑上色後用 `transfer_painted_texture.py` 做混合貼圖 |
| 相機擬合時 CUDA 記憶體不足 | 一次最多跑 3 個 `prep_09b` |
| Mac 跑到一半停了 | 用 `caffeinate -is` 防止休眠 |
| Stage 2 在 GPU 上很慢 | 物理是 float64；用 CPU，或用 `--stage2_E stage2.npz` 只模擬不最佳化 |
| UniRig 在 Mac 上報 `src must be CPU tensor` | 用 `prep_00_unirig_mac.sh`，它預設用 CPU |

## 11. 和論文不同的地方

| 項目 | 論文 | 這裡 | 原因 |
| --- | --- | --- | --- |
| 3D 模型 | 原本的模型 | 從影片第 0 幀用 Hunyuan3D-2 重建 | 論文沒公開模型 |
| 影片 | Kling 生成 | 論文網頁上的參考影片；或本機 LTX-2.3 | 這樣才能和論文直接比 |
| 相機 | 已知 | 自己擬合 | 網頁沒給 |
| 物理引擎 | NVIDIA Warp | PyTorch（float64） | 方程式相同，已用有限差分驗證 |
| 四面體網格 | TetWild | fTetWild（`pytetwild`） | 同一系列的後續版本 |
| 蒙皮權重 | UniRig 原樣 | 選用平滑 `--skin_smooth` | UniRig 的權重會在一條邊上換骨頭 |
| 邊長項 | 無 | 選用 `--w_rigid` | 末端關節亂轉會把臉扯皺 |
| 去拉扯 | Stage 2 | 另外加 `arap_refine.py` | 去得比 Stage 2 多，只要幾十秒 |

論文沒公開 loss 權重、迭代數、Eq. 9 的門檻、Eq. 8 的深度正規化、楊氏模數初值等，這裡都是自己假設的，結果會受影響。完整清單見 [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md#values-the-paper-does-not-publish)。

## 授權與致謝

- 論文：Tianyi Xie 等人，*AniMimic: Imitating 3D Animation from Video Priors*，CVPR 2026。參考影片和論文結果影片屬於原作者，不包含在這個 repo 裡（用 `download_paper_refs.sh` 下載）。
- Hunyuan3D-2 依騰訊的 Tencent Hunyuan 非商業授權使用，用它重建的模型和貼圖也受這個授權約束。
- 現成骨架模型（CC BY 4.0）：Cow NPC（nataliekirk）、熊（Nyilonelycompany）、大象（shahabahmed）、麋鹿（dinomaster），詳見 `data/hf_*/CREDIT.md`。
- UniRig、SAM2、VGGT、AllTracker、LTX-2 各依其原始授權。
- `lida/` 的程式碼還沒有選定授權。
