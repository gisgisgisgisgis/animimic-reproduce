# AniMimic reproduction

將現成 rigged 3D mesh 的動作對齊到 I2V 生成的參考影片，並用可微分 FEM 做第二階段 refinement：

```text
rigged GLB → canonical image → Wan / LTX 影片
           → SAM2 mask + AllTracker tracks + VGGT depth
           → joint optimization → fTetWild → FEM material optimization
           → GLB / NPZ / MP4
```

不需要訓練模型。每段參考影片與 supervision 都會存檔、快取。
第一次使用建議先跑 **LTX 2B 的單獨影片生成**，成功後再安裝完整 pipeline。

## 直接執行 sh 腳本（建議入口）

已把步驟拆成多個 Bash 腳本。在 GPU 主機同步／clone 完專案後，先進入目錄：

```bash
cd ~/animimic-reproduce
bash scripts/01_create_env.sh
bash scripts/02_install_video.sh
```

`01` 會建立 repo 內的 Python 3.11 `.venv`；若已存在相容環境就沿用，
不會刪除既有環境。它優先使用系統 `python3.11`，否則使用 Conda。
如果兩者都沒有，先執行 `bash scripts/00_install_miniforge.sh`，再重跑 `01`。
後續腳本會自動使用此環境，不必手動 activate，也不會使用系統 Python 3.8。

只生成 LTX 影片，準備自己的圖片後執行：

```bash
bash scripts/07_generate_video.sh inputs/reference.png configs/fox-ltx.yaml runs/reference-ltx.mp4
```

完整 pipeline，接著依序執行：

```bash
# 若主機找不到 nvcc，先安裝使用者目錄內的 toolkit：
bash scripts/12_install_cuda_toolkit.sh
bash scripts/03_install_pipeline.sh ltx
bash scripts/05_fetch_assets.sh horse_textured
# 先看輸入圖，再只生成影片；不會接著跑 joint/FEM：
bash scripts/13_review_video.sh --render-only
bash scripts/13_review_video.sh
# 確認影片合格後才繼續：
bash scripts/04_build_tetwild.sh
bash scripts/06_check.sh configs/horse-textured-ltx-5s.yaml
bash scripts/08_run_pipeline.sh configs/horse-textured-ltx-5s.yaml
```

`13` 預設使用有 2K 毛色貼圖的新馬（`horse_textured`），生成約 5 秒影片
（121 幀、24 fps），輸出位置會印在終端機：
`runs/horse_textured/<run-id>/canonical.png`、`reference.mp4` 與 `video-review.jpg`。
先確認只有一匹馬、動作確實發生、沒有整隻滑走、外形和鏡頭穩定，再執行完整流程。
若影片不合格，修改 `configs/horse-textured-ltx-5s.yaml` 的 prompt 或 seed 後重跑 `13`。
確認後用同一份 YAML 跑 `08`，會共用影片快取，不需重新生成。

舊的 `horse` 是純色材質的帶鞍模型，沒有毛色貼圖細節，保留作流程除錯；
不能將其色塊外觀視為寫實動物輸入。影片即使外形穩定，若只有運鏡、
沒有指定的局部動作，也不合格，應停在影片檢查階段。

不要在前一個步驟失敗時直接繼續下一個。腳本不會使用 sudo 安裝系統驅動，
完整 pipeline 仍需主機提供相容的 CUDA compiler／C++ compiler。
若 `01` 建立的是 Conda prefix，`04` 會在該環境安裝 CMake/GMP；
使用一般 venv 時，CMake/GMP/C++ compiler 需由系統提供（見 6.1）。
`03` 使用 `scripts/runtime-constraints.txt` 防止上游依賴靜默替換固定版本；
若版本衝突，pip 會報錯並停止。

| 腳本 | 用途／預設 |
| --- | --- |
| `00_install_miniforge.sh` | 選用；安裝到 ~/miniforge3 |
| `01_create_env.sh` | 建立／檢查 Python 環境 |
| `02_install_video.sh` | CUDA PyTorch 2.6 + LTX 依賴 |
| `03_install_pipeline.sh [ltx\|wan]` | supervision、PyTorch3D、FEM 依賴；預設 ltx |
| `04_build_tetwild.sh` | 編譯 fTetWild |
| `05_fetch_assets.sh [角色 ...]` | 下載指定角色；預設 horse |
| `06_check.sh [CONFIG]` | 完整環境 preflight |
| `07_generate_video.sh IMAGE [CONFIG] [OUTPUT]` | 單獨生成／重用快取影片 |
| `08_run_pipeline.sh [CONFIG] [RUN FLAGS ...]` | 完整 pipeline；預設 horse-ltx |
| `09_download_wan.sh` | 選用；下載 Wan 5B 權重 |
| `10_tests.sh` | offline tests + pip check |
| `11_preview_asset.sh [角色]` | CPU 多視角資產預覽；預設 horse |
| `12_install_cuda_toolkit.sh` | 選用；在 .cache/cuda-12.4 安裝 nvcc / CUDA toolkit，不更動 driver |
| `13_review_video.sh [CONFIG] [--render-only]` | 預設 horse-textured-ltx-5s；先檢查輸入與影片，停止在 render 或 video |

每個入口都可用 `--help` 查看參數。例如：

```bash
bash scripts/08_run_pipeline.sh --help
bash scripts/05_fetch_assets.sh horse puppy shark
bash scripts/08_run_pipeline.sh configs/horse-ltx.yaml --from-stage joints
bash scripts/11_preview_asset.sh puppy
bash scripts/10_tests.sh
```

切換 Wan：

```bash
bash scripts/03_install_pipeline.sh wan
bash scripts/09_download_wan.sh
bash scripts/06_check.sh configs/fox.yaml
bash scripts/08_run_pipeline.sh configs/fox.yaml
```

腳本自動設定 HF/Torch cache 到 repo 的 `.cache/`，也會自動加入 fTetWild 的 PATH。
若要沿用已建立的其他環境，將 `ANIMIMIC_ENV_DIR` 指向該環境根目錄。
GPU 編號、下載快取位置和編譯平行度都可由外部設定，既有設定會保留：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/08_run_pipeline.sh configs/horse-ltx.yaml
MAX_JOBS=1 bash scripts/03_install_pipeline.sh ltx
# 只有在使用其他已存在的 Python 環境時才需要：
ANIMIMIC_ENV_DIR="$HOME/miniforge3/envs/animimic" bash scripts/06_check.sh
```

腳本可從其他工作目錄以絕對路徑呼叫；自己傳入的相對檔案路徑，
依「呼叫時的工作目錄」解析。以下各節保留逐步指令，方便了解與排錯。

## 1. 系統與版本

以下指令以 **Linux x86-64、Bash、NVIDIA CUDA GPU** 為例；Windows 請在
有 GPU 支援的 WSL2 內操作。在遠端伺服器上，安裝和執行指令都要在該伺服器的
終端機執行，而不是在自己的筆電另建一份環境。

| 項目 | 使用版本／用途 |
| --- | --- |
| Python | 建議 **3.11**；程式要求至少 3.11，不能使用 3.8 |
| PyTorch | **2.6.0** |
| torchvision | **0.21.0**，搭配上述 PyTorch |
| CUDA wheel | 本文使用 **cu124**；驅動需相容 CUDA 12.4 |
| CUDA toolkit / nvcc | 完整 pipeline 編譯 PyTorch3D 時需要，本文搭配 **12.4** |
| Diffusers | **0.35.1** |
| Transformers | **4.51.3** |
| Accelerate | 套件要求 >=1.2、<2；本地使用 1.10.1 |
| Warp | **1.19.0**，FEM backend |
| PyTorch3D | bootstrap 固定在 `3eac7c62fed39a09768cd08119c5e1bf4daef559` |
| fTetWild | 由官方 source 編譯，執行檔為 `FloatTetwild_bin` |

版本來源見 [pyproject.toml](pyproject.toml)；
[requirements-tested.txt](requirements-tested.txt) 記錄數值測試環境，
不是包含所有上游模型的完整 CUDA lockfile。

先確認主機看得到 GPU：

```bash
nvidia-smi
```

`nvidia-smi` 顯示的是 driver 支援的 CUDA 版本；
`torch.version.cuda` 是 PyTorch wheel 的版本；
`nvcc --version` 是編譯工具版本，三者不是同一件事。
只生成影片不需要 nvcc，編譯 PyTorch3D 才需要。

沒有 sudo 的實驗室主機，驅動與編譯工具請使用管理員提供的環境／module，
不要自行更動系統 driver。若有多張 GPU，依實驗室排程規則選擇可用卡，
例如 `CUDA_VISIBLE_DEVICES=0 animimic ...`。

## 2. 下載專案

```bash
git clone https://github.com/gisgisgisgisgis/animimic-reproduce.git
cd animimic-reproduce
```

後面的相對路徑都以此目錄為準。已經 clone 的人只需進入原目錄，
有需要時執行 `git pull --ff-only`，不要在別的專案裡安裝。

## 3. 建立虛擬環境：二選一

不要同時疊加 Conda 和 venv。下面兩條路徑擇一完成即可。

### A. Conda / Miniforge（系統 Python 太舊時建議用這個）

已有 conda 可直接建立環境：

```bash
conda create -n animimic python=3.11 pip -y
conda activate animimic
python --version
python -m pip --version
```

預期 Python 顯示 `3.11.x`，pip 路徑也應位於 `animimic` 環境。

如果沒有 conda，可先依 [Miniforge 官方說明](https://github.com/conda-forge/miniforge)
安裝。Linux x86-64 的一種做法如下；安裝路徑若已存在，直接使用原有安裝：

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -n animimic python=3.11 pip -y
conda activate animimic
```

### B. Python venv（已經有 Python 3.11）

先檢查 interpreter；找不到 `python3.11` 時，請改用上面的 Conda 路徑。
Ubuntu 的 Python 3.11 若缺 venv 模組，需要先安裝對應的
`python3.11-venv` 套件，或請管理員安裝。

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip --version
```

`python3 -m venv` 會沿用系統的 Python；
若系統是 3.8，建立出的環境仍是 3.8，不會自動變成 3.11。
已有舊版環境時，請另取新環境名稱或使用 Conda，不要沿用舊環境。

每次重新開終端機，都要重新 `conda activate animimic`，
或在 repo 內 `source .venv/bin/activate`。

## 4. 安裝 CUDA PyTorch 與專案

在剛建立的環境裡執行：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

預期 torch 為 `2.6.0+cu124`、wheel CUDA 為 `12.4`、
GPU available 為 `True`。若為 False，先處理 GPU/driver/容器配置，
不要繼續下載大型權重。其他 CUDA wheel 組合請對照
[PyTorch 官方舊版安裝表](https://pytorch.org/get-started/previous-versions/)。

## 5. 路徑一：先只生成一段 LTX 影片

這條路徑不需要 SAM2、AllTracker、VGGT、PyTorch3D 或 fTetWild。

```bash
python -m pip install -e '.[video]'
python -m pip check
animimic --help
```

準備一張自己的 PNG/JPG，例如 `inputs/reference.png`。
路徑需改成你實際存放的圖片：

```bash
animimic video --config configs/fox-ltx.yaml --image inputs/reference.png --output runs/reference-ltx.mp4
```

首次執行自動下載 `Lightricks/LTX-Video` 的原始 2B 權重及 text encoder。
預設 512×288、49 幀、30 steps，並啟用 CPU offload。
offload 使用 CPU RAM 暫存部分模型，生成仍使用 GPU。

輸出為 `runs/reference-ltx.mp4`。相同圖片、prompt、seed 與設定重跑會讀快取。
範例 prompt 寫的是 fox；換成自己的圖片時，請複製 YAML 並修改
`video.prompt`，讓文字描述符合圖片。

## 6. 路徑二：完整 3D animation pipeline

先完成第 1–4 節，再做本節。若已安裝 video extra，可以直接繼續。

### 6.1 系統工具與 CUDA compiler

Ubuntu/Debian 有 sudo 時可安裝一般編譯工具：

```bash
sudo apt update
sudo apt install -y git build-essential cmake libgmp-dev ffmpeg
```

沒有 sudo 時，請使用主機已提供的 compiler、CMake、GMP。
Conda 使用者也可在自己的環境安裝 CMake/GMP：

```bash
conda install -c conda-forge 'cmake=3.29.*' gmp -y
```

接著確認 CUDA toolkit：

```bash
nvcc --version
which nvcc
```

對本文 cu124 路徑，請使用 CUDA toolkit 12.4。
若 `03_install_pipeline.sh` 顯示 `nvcc missing`，可直接執行：

```bash
bash scripts/12_install_cuda_toolkit.sh
bash scripts/03_install_pipeline.sh ltx
```

這會透過 Conda 在 repo 的 `.cache/cuda-12.4` 安裝 NVIDIA
CUDA toolkit 12.4.1，不需要 sudo，也不會替換系統驅動或 Python 環境。
後續 `03` 會自動找到它；下載可能需要數 GB 空間。
需先有 Conda（沒有時可使用 `00_install_miniforge.sh`）。
官方安裝來源：[NVIDIA CUDA 12.4.1 Linux guide](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html)。

若伺服器提供 environment modules，先用 `module avail cuda` 找可用版本，
再依主機實際名稱載入。若已裝在 `/usr/local/cuda-12.4`，可以設定：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

路徑只是一個範例；請使用主機實際的 toolkit 路徑。
CUDA compiler 對 GCC 版本有要求，請搭配該 toolkit 支援的 compiler；
編譯失敗時不要直接加 `--allow-unsupported-compiler` 略過檢查。

### 6.2 Python 與上游模型套件

```bash
python -m pip install -e '.[gpu,video]'
MAX_JOBS=2 python scripts/bootstrap.py --video-backend ltx
python -m pip check
```

bootstrap 會 clone 並安裝 AllTracker、SAM2、VGGT、PyTorch3D，
clone fTetWild source，並記錄 Git revision 到 `third_party/revisions.json`。
LTX 路徑不 clone Wan；已存在的 Wan 目錄與版本記錄會保留。

PyTorch3D 需要 C++/CUDA 編譯，第一次安裝較久。
記憶體不足時把 `MAX_JOBS=2` 改成 `MAX_JOBS=1`。
上游 requirements 會隨版本變動；若 pip 報衝突，保留錯誤訊息，
不要任意升級本專案固定的 torch/torchvision。
安裝後再檢查一次：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "from pytorch3d import _C; from sam2.sam2_video_predictor import SAM2VideoPredictor; from vggt.models.vggt import VGGT; print('imports OK')"
```

### 6.3 編譯 fTetWild

在 repo 根目錄執行：

```bash
bash scripts/04_build_tetwild.sh
export PATH="$PWD/third_party/fTetWild/build:$PATH"
FloatTetwild_bin --help
```

腳本會在 Conda 環境中安裝 CMake 3.29.x，避免新版 CMake 的 CMP0169
政策與 libigl 使用的 `FetchContent_Populate` 衝突。若使用 Python venv，先執行
`python -m pip install 'cmake>=3.29,<3.30'`，再跑腳本。

預期最後能列出 fTetWild CLI 的參數。如果 CMake 找不到 GMP，
請確認已裝 `libgmp-dev`；Conda 環境可嘗試在 configure 指令追加
`-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"`。
編譯方式可參考 [fTetWild 官方文件](https://github.com/wildmeshing/fTetWild)。

上述 PATH 設定只對目前終端機有效，每次重新登入需要再設定。
也可在 YAML 指定執行檔的絕對路徑：

```yaml
physics:
  tetwild_executable: /absolute/path/to/FloatTetwild_bin
```

### 6.4 下載一個動物並檢查環境

```bash
animimic assets list
animimic assets fetch --asset horse
animimic inspect assets/downloads/horse/model.glb
animimic assets preview --asset horse --output runs/asset-previews
animimic preflight --config configs/horse-ltx.yaml
```

資產會存到 `assets/downloads/horse/`，包含 GLB、hash 與 attribution manifest。
preview 輸出 `runs/asset-previews/horse.png`。
preflight 會檢查完整 pipeline 的套件、repo、CUDA 與 fTetWild；
有任何 missing 時先修正，再執行完整流程。

### 6.5 開始執行

```bash
animimic run --config configs/horse-ltx.yaml
```

終端機會列出目前階段、是否使用快取，以及各最佳化階段的迭代進度，例如：

```text
[stage 4/8 supervision] Cache hit | elapsed 00:00:01
[stage 5/8 joints] Running
[joints 1/3 root] 1/100 (1.0%) | loss=0.42 | elapsed 00:00:08 | ETA 00:13:12
```

ETA 依目前階段已完成的迭代估算，不包含後續階段。較長的迭代會間隔顯示
正在渲染或模擬的影格。FEM 會立即列出初始姿勢求解；內部 Newton、CG、
步長搜尋與反向傳播的 adjoint 求解也會間隔輸出目前迭代與殘差。
這些是實際運算點的更新；單次 GPU 操作尚未完成時，可能暫時沒有新訊息。

首次執行會依序下載／執行所需權重：

| 模組 | 預設權重來源 | 下載方式 |
| --- | --- | --- |
| LTX | `Lightricks/LTX-Video` | 首次推論自動下載 |
| SAM2 | `facebook/sam2.1-hiera-large` | 首次 supervision 自動下載 |
| AllTracker | `aharley/alltracker/alltracker.pth` | 首次 tracking 自動下載 |
| VGGT | `facebook/VGGT-1B` | 首次 depth inference 自動下載 |
| Wan | `Wan-AI/Wan2.2-TI2V-5B` | 改用 Wan 時手動下載，見下一節 |

權重可能佔用大量磁碟空間。預設 Hugging Face / Torch cache 通常在家目錄；
若家目錄配額小，可在執行前改放到有空間的工作磁碟，例如：

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
```

這兩個變數也需要在重新登入後重新設定。
一般公開下載不需要 token；若 Hugging Face 回傳存取限制，
請依相應模型頁完成帳號／授權設定。

## 7. 改用 Wan，或切回 LTX

Wan 的安裝與權重保留為獨立路徑：

```bash
MAX_JOBS=2 python scripts/bootstrap.py --video-backend wan
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan2.2-TI2V-5B
animimic preflight --config configs/fox.yaml
animimic run --config configs/fox.yaml
```

`huggingface-cli` 由此環境的 huggingface-hub 套件提供。
需要另一個權重／repo 路徑時，可修改 YAML 的 `video.checkpoint`、
`video.repo`。預設 Wan 是 1280×704、121 幀，使用 model offload 與 CPU T5。

也能用 CLI 切換：

```bash
animimic run --config configs/fox.yaml --video-backend ltx
```

切換 backend 時會保留 prompt/seed，重設另一個模型不能沿用的參數。
要精確控制解析度、幀數、steps，請編輯 YAML。
完整說明見 [本地影片模型與動物指南](docs/local-video.md)。

## 8. 資產、設定與自己的影片

目前共九個資產：Fox、humanoid、robot、兩個馬模型，以及狗、鯊魚、海鷗、海龜。
Fox 與舊馬保留作外觀除錯；動物的來源、面數、骨架數、
授權與實際預覽都在 [資產指南](docs/local-video.md)。
每次只需下載要用的角色，不必下載整套 Objaverse。

例如改跑狗：

```bash
animimic assets fetch --asset puppy
animimic run --config configs/puppy-ltx.yaml
```

自訂 GLB 或重用現成影片，可在複製的 YAML 加上：

```yaml
asset: my_character
asset_path: /absolute/path/to/rigged-model.glb
reference_video: /absolute/path/to/reference.mp4
video:
  backend: ltx
  fps: 24
  sample_frames: 25
```

GLB 必須有 mesh、skeleton、inverse bind matrices 與 skinning weights。
`reference_video` 只跳過生成，後續 supervision / optimization 仍使用 GPU。
影片與 mesh 應是同一角色、相近起始姿勢與固定視角。
若要調整自訂模型朝向，可設定 `asset_rotation: [-90, 0, 0]`
（XYZ 角度）；請先看 preview 確認實際朝向。

## 9. 輸出、快取與重跑

每次 run 的 manifest 在 `runs/<asset>/<run-id>/manifest.json`，
實際 stage artifacts 在 `.cache/animimic/<stage>/<key>/`。
manifest 會列出各 stage 的目錄與 SHA-256。

```bash
animimic inspect runs/horse/你的-run-id
```

請將上面的 run-id 換成實際目錄名稱。主要輸出：

| artifact | 用途 |
| --- | --- |
| `reference.mp4` | 原始 I2V 參考影片 |
| `supervision.npz` | frames、mask、tracks、confidence、depth |
| `joint_trajectory.npz` / `joint.glb` | 骨架動畫 |
| `physics_trajectory.npz` | FEM tet/surface trajectory 與材質參數 |
| `joint_losses.*` / `physics_losses.*` | JSON / CSV loss history |
| `joint_0.mp4` / `physics_0.mp4` | 與 reference 並排的 input-view 預覽 |
| `joint_-45.mp4`、`physics_45.mp4` 等 | novel-view 預覽 |
| `metrics.json` | mask IoU、tracking error、vertex acceleration |

同設定重跑會檢查 cache；調整 joint/FEM 設定不需要重新生成相同影片。
若已有前段 cache，可從指定 stage 開始：

```bash
animimic run --config configs/horse-ltx.yaml --from-stage joints
```

此指令要求相同設定的前置 cache 存在；若缺少，請先用一般 `run`。
刻意要重算某個 stage 時才使用 `--force-stage supervision` 或
`--force-stage video`，因為它會重寫該 stage 的快取檔案。

`Animimic.pdf`、GLB、權重、cache 與 run artifacts 都在 Git ignore 範圍，
不會隨一般程式 commit 上傳；文件內的小型預覽圖例外。

## 10. 測試與常見問題

在完成對應依賴安裝的環境執行：

```bash
python -m unittest discover -s tests -v
python -m pip check
```

測試包含 FK/LBS、GLB round-trip、真實 PyTorch3D rendering、
小型隨機 LTX I2V、Warp/PyTorch 能量與梯度比對，以及 synthetic joint/material recovery。
小型 LTX 測試使用隨機權重，不會下載正式 2B 模型。
缺少 optional 依賴時部分測試會 skip；不能把 skip 當作完整安裝通過。

| 現象 | 處理方式 |
| --- | --- |
| Python 是 3.8 / 找不到可安裝版本 | 回第 3 節，用 Python 3.11 重建新環境 |
| `animimic: command not found` | 啟用正確環境，從 repo 執行 `python -m pip install -e '.[video]'` |
| `CUDA unavailable` | 檢查 `nvidia-smi`、排程分配、CUDA_VISIBLE_DEVICES 與 CUDA PyTorch wheel |
| PyTorch3D 編譯失敗 | 檢查 torch/torchvision、nvcc/toolkit、GCC 相容性；用 MAX_JOBS=1 降低記憶體 |
| Wan 安裝 flash_attn 時顯示 `No module named 'torch'` | 更新程式後重跑 `03_install_pipeline.sh wan`；腳本會先裝建置工具，再以 `--no-build-isolation` 使用現有 PyTorch 建置 |
| `FloatTetwild_bin` 找不到 | 完成 6.3 並設定 PATH 或 physics.tetwild_executable |
| GPU OOM | 先用 LTX；降低 video.width/height（32 的倍數）、num_frames（8k+1） |
| 權重下載空間不足 | 把 HF_HOME / TORCH_HOME 設到有空間的磁碟 |
| source requirements 衝突 | 記錄錯誤與第三方 Git revisions；不要直接把固定版本全升到最新版 |

`flash_attn` 的建置會載入 PyTorch；pip 的隔離建置環境可能找不到目前 Conda
環境裡的 torch。Wan 安裝流程依 [FlashAttention 官方安裝方式](https://github.com/Dao-AILab/flash-attention#installation-and-features)
停用 build isolation，先安裝 setuptools、wheel、packaging、psutil 與 ninja。
遇到這個錯誤不需要重建環境。保留 `03` 的版本約束與 `MAX_JOBS` 設定，
更新後直接重跑即可；若主機記憶體不足，可設定 `MAX_JOBS=1`。

## 方法設定

Joint stage 使用 6D rotation、FK/LBS 與 RGB、mask、tracking、normalized depth、
temporal smoothness、joint regularization losses。
FEM 使用 fixed-corotated elasticity、backward Euler、Newton/CG、
joint boundary conditions 與 implicit adjoint；Warp 計算能量，
PyTorch 提供相同公式的導數與 Hessian-vector products。

論文未公開的超參數採用本專案 defaults：Adam 1e-3、loss weights 1、
25 個 sampled frames、Poisson ratio 0.45、normalized density 1、四個 substeps。
材質數值對應 normalized mesh 尺度。無 gravity 或 collision/contact。
`horse_textured` 的輸入與渲染保留完整 UV 貼圖；其他預設與 skeletal GLB 匯出
仍使用 vertex colors。FEM deformation 存 NPZ，
skeletal GLB 只包含 joint animation。
