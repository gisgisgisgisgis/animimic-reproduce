# 兩套實作的對照

上層的 `src/animimic` 和這個資料夾 `lida/` 都在重現 AniMimic（CVPR 2026），但目標和做法不同，兩邊可以互補。上層的內容以 2026-11-19 的 `main`、`master`、`fix/joint-temporal-stability` 為準。

## 目標

- **上層 `src/animimic`**：論文的使用情境。現成綁好骨架的動物 → 渲染 → 生成影片（LTX 2B 或 Wan 5B）→ 關節最佳化 → FEM → 匯出 GLB。有完整的 CLI、YAML 設定和每一步的快取。
- **`lida/`**：和論文的結果直接比較。用論文網頁上的 8 段參考影片，網頁沒有附模型，就從影片第 0 幀重建，再在同一段影片上和論文自己的結果比分數。另外也做了現成骨架模型加上 Mac 上的 LTX-2.3。

## 逐項對照

| 部分 | 上層 `src/animimic` | `lida/` |
| --- | --- | --- |
| 平台 | Linux + NVIDIA GPU；沒有 CUDA 時 GPU 的步驟會停下 | Mac（Apple Silicon，MPS）和 Linux + NVIDIA GPU；Stage 2 的物理在 Mac 上用 CPU |
| 輸入 | Objaverse 的動物 GLB（`animimic assets fetch`）加上生成的影片 | 論文網頁的影片（`scripts/setup/download_paper_refs.sh`）；或現成骨架 GLB 加 LTX-2.3 |
| 模型與骨架 | 資產自己的骨架和權重 | 論文影片：Hunyuan3D-2 從第 0 幀重建 → 擬合相機和視野角 → UniRig 自動骨架。現成資產：自己的骨架 |
| 相機 | 設定檔裡固定的方位角、仰角、距離 | 從第 0 幀的輪廓擬合（網格搜尋 + 可微分細調） |
| 渲染 | PyTorch3D：顏色用硬邊緣（blur 0、每像素 1 面），輪廓用 SoftSilhouette（blur 1e-4、每像素 10 面） | 純 PyTorch 的軟光柵（`animamimic/render.py` 的 `RasterRenderer`），顏色和輪廓用同一個 sigma（1e-5，384 px） |
| 取樣 | 25 格，每一步畫全部的格 | 32 格，每一步隨機畫 4 格 |
| 關節最佳化 | `main`：root 100 步 → geometry 300 步 → full 300 步；Huber、BCE + IoU 輪廓、以中位數和 MAD 標準化深度；旋轉上限 75°、位移上限 0.1 倍骨長；梯度裁切 1；權重都是 1。`fix/joint-temporal-stability`：改成單一階段，照論文 Eq. 5–9 用每格的 L2 和逐分量裁切 | 單一階段 3000 步；L1；每格標準化深度；Eq. 9 逐分量門檻（`--trans_max 0.01 --w_reg 1000`）；權重 track 10、smooth 10。另外加了論文沒有的 `--skin_smooth 30`（蒙皮權重平滑）和 `--w_rigid 10`（邊長項） |
| FEM | Warp 算能量、PyTorch 算導數。`main`：matrix-free CG；`master`：稀疏 Newton 加解析 Hessian。泊松比 0.45、4 個子步 | PyTorch float64 的 Newton + PCG，adjoint 梯度；泊松比 0.4、2 個子步；Newton 從蒙皮姿勢起步；E 由粗到細 10 + 10 次 |
| 去除表面拉扯 | — | ARAP（`scripts/arap_refine.py`），接在 Stage 1 之後 |
| 貼圖 | 資產原本的 UV 貼圖或頂點色 | 第 0 幀烘焙加左右對稱補色；Hunyuan3D-Paint 生成的「混合貼圖」 |
| 評估 | 輪廓 IoU、追蹤點誤差、頂點加速度 | 再加上論文 Table 1 的 SSIM、LPIPS，和論文結果並排的影片，表面拉扯數字與熱圖，VLM 逐格檢查 |
| 工程 | CLI、YAML、每個 stage 的快取與 manifest、編號腳本、單元測試 | 一步一個腳本；Mac 和伺服器分工的腳本；單元測試（`tests/test_core.py`，13 項） |

## 可以互相借用的地方

**上層 → `lida/`**

- 影片生成、快取和影片審查的流程。
- Warp 或稀疏 Newton 的 FEM：`lida/` 的 Stage 2 在 RTX 4090 上每次迭代要約 21 分鐘（float64 在遊戲卡上很慢）。

**`lida/` → 上層**

- **ARAP**：只需要靜止網格和每一格的頂點位置，可以直接接在 joints 之後當一個 stage。在白狼、小狗、貓上，表面最糟 1% 的拉扯從 13.9%／21.7%／20.3% 降到 6.3%／9.9%／8.8%，比論文的 Stage 2（9.4%／15.8%／11.4%）還多，輪廓幾乎不變。
- **和論文結果比較**：`tests/compare_with_paper.py`（SSIM、LPIPS、輪廓 IoU、五欄並排影片）、`tests/surface_distortion.py`（拉扯），可以加進 evaluate。
- **論文網頁影片的路線**：重建模型和擬合相機（`scripts/prep_09a_frame0_to_3d.py`、`scripts/prep_09b_fit_frame0.py`），可以當成一種新的輸入。
- **遇過的坑**（詳見 [docs/UNDERSTANDING.md](docs/UNDERSTANDING.md#一路上遇到的問題)）：
  - UniRig 的骨架和模型不在同一個座標系：要先把節點變換烘焙進模型。
  - 骨頭平移沒限制：Adam 會靠平移做動作，模型被撕開。
  - UniRig 的權重會在一條邊上換骨頭：交界處被拉開。
  - 鼻尖、耳尖這種末端關節：追蹤點太少，會亂轉把臉扯皺。
  - Stage 2 的時間步長要跟著影片：取 32 格時是 (32 − 1) ÷ 影片秒數，不是原本的 24 fps。
