# AniMimic 實作與輸入監督核對

依據本地 `Animimic.pdf` 的 §3.1.1–3.1.2、式 (3)–(9)、§4 Implementation Details
核對。PDF 只在本地讀取，不納入 Git。以下結論涵蓋 joint stage；不代表整套 FEM 已完成論文對齊。

## 依公式修改後的 joint 行為

| 項目 | 論文 | 現在的程式 |
| --- | --- | --- |
| Rig | UniRig 預測階層與 skinning weights | 依使用者選擇直接用已有 rig 的 GLB；保留此替代輸入 |
| I2V | Kling image-to-video；文獻 [50] 是 Kling 2.1 | 依使用者選擇保留 Wan2.2-TI2V-5B／LTX；不是 Wan-Animate，不能視為原文實驗設定 |
| 關節參數 | 每關節 6D rotation 與 translation，FK + LBS | 全部關節從第一次更新就一起最佳化。使用 GLB rest-local transform 上的增量參數 |
| RGB，式 (5) | 每影格 rendered RGB 與 reference 差值 norm，再平均 | 全畫面的逐影格 L2 norm 平均；不再用前景 Charbonnier |
| Mask，式 (6) | 每影格 silhouette 與 SAM2 mask 差值 norm，再平均 | 逐影格 L2 norm 平均；不再用 BCE + IoU |
| Tracking，式 (7) | barycentric 3D points 投影與 pixel trajectories 差值 norm | 逐影格 L2 norm 平均，使用 pixel 座標；不再除影像尺寸或加信心權重／Huber |
| Depth，式 (8) | 相機深度 Z 對上 N(D) | 只有目標套用固定於 canonical mesh 的 N；預測 camera depth 直接進 L2 殘差，不再逐幀標準化預測 |
| Smoothness | 頂點的一階與二階時間差分 | 保留兩者的平方平均；具體 norm／reduction 是實作假設 |
| Regularization，式 (9) | 非 root 的 theta 與逐分量 clip(theta) 差之 L2 norm | 對非 root 的 learned 6D rotation／translation 逐分量 clipping，再逐影格 L2 平均；不再用 rotation geodesic angle 或骨長比例懲罰 |
| 最佳化排程 | 整段關節參數最佳化，Adam 1e-3 | 預設單一 full objective，六項 loss 從第一步啟用；移除 root-only／延後 RGB／gradient clipping |
| material 的影像目標，式 (13) | 使用與 joint 相同的 L | 共用 video_loss，且繼承 joint 的 lambda 權重；joint regularization 對固定 joint 不產生 material 梯度 |

## 必須明列的實作假設

原文明確給出目標函數的結構，但不足以復原所有數值選擇。以下都不是作者公開參數：

- 式 (5)–(8) 沒有標 norm 種類，目前選 L2。每個影格先 flatten 後計算 Euclidean
  norm，再對影格平均，沒有改成 MSE、Huber 或全序列單一 norm。
- N(D) 沒有具體定義。目前將每幀 observed depth 的 median/MAD 映射到**固定 canonical
  mesh 採樣點**的 camera-depth median/MAD。即 `N(Dt) = (Dt-median(Dt)) /
  max(MAD(Dt), 1e-6) * MAD(Zcanonical) + median(Zcanonical)`。
  canonical 的統計只用第一幀在畫面內的採樣點，與當前最佳化姿勢無關且不傳梯度。
  這是專案對 N 的明確實作假設，**不是已取得作者定義**。
- Tracking/depth 只排除越過影像範圍、無可用影像深度的觀測；沒有 visibility 或
  confidence 的連續權重。這個缺失觀測處理也未在論文細述；不表示遮擋點一定正確。
- regularization 的預設 bounds：6D rotation 每分量 ±1、translation 每分量 ±0.1
  normalized model units。可在 `joints.rotation_bound`、`joints.translation_bound` 設定。
  clipping 作用在 GLB bind pose 上的 learned rest-local parameters。原文沒有公開 bounds。
- 六項 lambda 預設 1、25 sampled frames、700 iterations 是專案數值設定。
  Adam 1e-3 則是論文明載。影格取樣數／分辨率會改變 norm 尺度，需要記錄並比較。

```yaml
joints:
  iterations: 700
  learning_rate: 0.001
  rotation_bound: 1.0
  translation_bound: 0.1
  loss_weights:
    rgb: 1.0
    mask: 1.0
    track: 1.0
    depth: 1.0
    smooth: 1.0
    regularization: 1.0
```

舊 YAML 的 `root_iterations`／`geometry_iterations`／`full_iterations` 若仍存在，
只將已提供的值相加作為總次數，不再切換排程；`iterations` 優先。
舊 `max_rotation_deg`／`max_translation_bone_fraction` 已不再控制 regularization，
應換成以上明列的 bounds。新參數與 loss 不會沿用舊 joint cache；不要把兩種目標下的
loss 數值當成同尺度直接比較。影片與監督可以沿用相符快取。

額外 joint acceleration、stationary-point prior、child translation freezing 預設關閉。
FEM 的重力、boundary／embedding 等差異未在這次 joint 公式修改中完成對齊；仍不能
稱整套系統已逐式完整重現。後續若取得作者 N、bounds、lambda，可直接針對這些明列假設更新。

## 已確認並修正的座標問題

`Renderer.surface_samples` 在 rasterization 的像素中心取得 barycentric coordinates。
PyTorch3D 的 screen projection 用 `(x+0.5, y+0.5)`，AllTracker 的 pixel grid
用整數 `(x, y)` 代表同一中心。原程式直接比較兩者，在完全靜止且對應正確時，
兩軸仍各差 0.5 pixel。

`Renderer.project` 現在轉為追蹤器座標。測試在橫式、直式畫面重建 barycentric 3D
採樣點，再投影回 query pixels，誤差需小於 1e-4 pixel。
這消除系統性偏移，**不能單獨解釋或保證解決腿部逐幀抖動**。

## 預訓練模型 adapter 核對

- AllTracker：依[官方 demo](https://github.com/aharley/alltracker/blob/master/demo.py)，
  多幀輸出 flow 要加回 query pixel grid 才是 trajectory；目前程式的加法正確。
  [官方模型](https://github.com/aharley/alltracker/blob/master/nets/alltracker.py)
  已對 vis/conf 做 sigmoid，現有 adapter 沒有重複 sigmoid。
  本地舊資料兩者都在 [0,1]，沒有把 logit 當機率的跡象。
- VGGT：對照[官方預處理](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py)，
  現有 `pad` 模式的 518 resize、14 pixel rounding 與中心 padding 反轉一致。
  目前逐張推論，原文只說 per-frame depth，沒有指定 batch/context；不能認定逐張推論等同作者設定。
  反轉 padding 正確並不能保證跨影格 depth 的尺度穩定。
- SAM2：用 canonical render 的 mask 初始化第 0 幀，再傳播。必須檢查生成影片第 0 幀
  是否仍與輸入 mesh 對齊；原程式只檢查 mask 非空，不能排除錯物件、遮擋或前景只剩局部。
- AllTracker 用未壓縮 sampled RGB；SAM2/VGGT 讀 sampled JPEG。有影像壓縮差異，
  本次記錄此差異，沒有據此判定為抖動原因。

## 本地舊 horse 資料的實際檢查（修改公式之前的輸出）

資料：`supervision/5a28614ae9da74ee2bbd577f9f751a48ec2de501618f8d9dabaad0397e97cee0/`。
這是先前 LTX 舊馬案例，**不是後來 textured horse，也未確認是外部 joint_0.mp4 的來源**。
另一份 `603283bf…` 的 tracking/depth 相同，mask 略有差異。

- 25 取樣影格、512 tracks。tracks / visibility / confidence / depth 都是有限值。
- SAM2 mask 第一／最後影格：5,348 → 2,352 pixels，面積剩 44%。
- 全部影格所有點相對第一幀的位移中位數 0.258 pixel、90 百分位 0.607 pixel。
- visibility × confidence > 0.5 的比例 65.5%；其中約 95.2% 落在當幀 mask 內。
  **這個高比例不等於追對有動作的馬**：實際疊圖顯示 tracker 留在中央原本靜止的深色馬，
  另一匹生成的棕馬從左側走進並遮住它。mask 主要追中央馬剩餘可見部分。
- 每幀 tracked depth 的 median 約由 0.919 到 1.002；MAD 跨影格範圍
  0.00111–0.02359。MAD 很小的影格，median/MAD normalization 會放大小差異。
  這是需要檢查的數值敏感性，尚未透過受控實驗證明是抖動主因。

同案例完整 joint history `24fbb2e1…` 的實際 loss：

| 時點 | mask | track | depth | vertex smooth |
| --- | ---: | ---: | ---: | ---: |
| root 起始 | 0.3933 | 0.000000642 | 1.5298 | 0 |
| full 最後 | 0.1960 | 0.001255 | 0.2515 | 0.004504 |

舊設定各項權重為 1。舊 Tracking 在像素正規化與 Huber 後數值極小，而 depth/mask
大得多；最後 tracking 反而變差。這支持「先檢查 supervision 與尺度」的優先順序，
但 loss 數值本身不等於梯度貢獻，不能只按數值倒數自動調權重。

## 如何檢查新影片的監督

不需 GPU，也不會重跑影片或更動監督。先從該 run 的 `manifest.json` 中
`stages.supervision.directory` 找出正確的 cache directory，接著執行：

```bash
conda activate animimic
export ANIMIMIC_ENV_DIR="$CONDA_PREFIX"
bash scripts/14_review_supervision.sh \
  .cache/animimic/supervision/實際的快取目錄/supervision.npz \
  runs/my-supervision-review
```

輸出 `supervision-review.jpg`（原影片／mask + tracks 疊圖）、JSON 與逐幀 CSV。
綠色是 mask、青色是可信 tracks、紅色是其他 tracks；可信門檻只用於診斷，
不會改動最佳化 loss。檢查是否追同一匹馬、頭腳是否都有點、遮擋後的點是否失效，
再看 mask 面積、in-bounds / mask coverage、depth median/MAD 是否突變。
這些檢查不使用「靜止點 3D 必須不動」的假設。

目前本地沒有新影片對應的完整 supervision，因此外部 joint_0.mp4 的確切成因仍待
該次 run 的 supervision.npz、joint_losses.json、config 與 reference.mp4 交叉核對。
