# Joint 預覽與實驗設定

## 按骨架部位取點與檢查追蹤

抬前腳實驗可先跑到 supervision，在 GPU 主機執行

```bash
bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-leftforeleg-grouped-tracks.yaml --until-stage supervision --debug-images
```

新設定使用同一份影片設定與 512 點預算，依可見表面的蒙皮權重分配給各部位。
同一部位內由中央開始，反覆選離已選點最遠的可見像素。取不到點的部位會留空，
其額度分給其他可見部位。設定中的骨架索引適用於目前的 `horse_textured` GLB。

終端機會印出 `supervision-review/` 的位置。裡面每個 `tracks-XXXXX.png` 對應
一個來源影格，顏色代表點最初對應的部位。實心表示 visibility × confidence > 0.5
且位置落在 SAM2 遮罩內。這只用來篩查，仍要目視確認點是否跟在正確的腳上。
`sam-mask-XXXXX.png` 與 `vggt-depth-XXXXX.png` 使用同一批來源影格編號。
`--debug-images` 預設關閉，開啟後可以直接從 supervision 快取產圖。
`track-parts.csv` 記錄各部位在每幀的點數。看完後用同一份設定加
`--from-stage joints --until-stage joints` 重用追蹤資料進行骨架擬合。

## 正規化關節擬合比較

新增 `joints.objective: normalized`，使用正規化 loss、較緊的位移限制與影格隨機取樣。
沿用本專案的 FK/LBS、既有蒙皮權重、PyTorch3D renderer 和像素座標。
沒有設定 `objective` 時仍使用原本的 `paper` 模式。這組實驗沒有更換模型、骨架或
renderer，也不包含 ARAP，實際馬影片的抖動改善仍需重跑確認。

以下兩份新設定與既有 balanced 實驗的 asset、camera、video 設定完全相同，
取樣維持 25 幀，讓相同 reference video／supervision 可以沿用。先跑到 joints：

```bash
# 原本模式，作為比較基準
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-joint-balanced.yaml --until-stage joints
# 正規化 loss 與姿勢限制
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-normalized.yaml --until-stage joints
# 相同正規化設定，再加邊長約束
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-normalized-rigid.yaml --until-stage joints
```

這兩份新設定使用 3000 步 Adam、lr 0.001、每步隨機渲染 4 幀（seed 0）。
Tracking、depth 和頂點時間平滑仍使用全部取樣幀。具體差異：

- RGB／mask 使用平均絕對誤差；參考 RGB 依 SAM2 mask 合成白底。
- Tracking 用有效點的平均 2D 距離，再除以影像長邊。使用 AllTracker
  channel 1 > 0.1 門檻（本專案的 `confidence`），第一幀保留在畫面內的採樣點。
  本專案 renderer 已轉為 tracker 座標，不需要額外的 +0.5 pixel 偏移。
- 深度按每幀有效點的 mean/std 對齊固定 canonical 深度，空幀不計 loss；單點幀也保持有限值。
- 頂點一、二階差分用 xyz 平方和再平均，權重 10。Tracking 權重也為 10。
- 非 root 的旋轉限制作用在相對 identity 的 6D 增量；旋轉／位移超界量使用平方懲罰。
  translation bound 0.01、regularization 權重 1000，是軟性懲罰，不是強制鎖住骨頭。
- `loss_weights.rigid: 10` 啟用相對靜止邊長的平方懲罰，0 則關閉。
  使用不重複的非退化邊；只對分母加下限，避免極短邊主導，不改動其靜止目標邊長。

新增的 `joint-review/joint_diagnostics.json` 記錄平均邊長變形、逐幀 p99 變形的平均、
頂點二階差分，以及按秒計算的加速度平方平均。只在相同模型、影片、取樣下比較，
數值越低不一定越好：也要看 `joint_0.mp4` 的動作是否跟上影片，與斜側面是否正常。
兩個新設定沒有啟用 ARAP、蒙皮權重平滑、靜止點 prior 或額外關節加速度 loss。
若繼續跑 FEM，影像／tracking／depth／smooth 目標會沿用所選 objective；邊長項只用於 joints。

只修改關節目標時 joint cache 會重算，前段影片／supervision 的 cache key 不受影響。
執行時依 manifest 檢查實際使用的影片；不要刪除 cache，也不需要 `--force-stage video`。

### Wan 單一動作比較

從 `master` 的 `93c9093` 搬入三份影片設定，另各提供一份 `normalized-rigid` 關節設定：

| 動作 | 原設定檔 | 新關節設定檔 |
| --- | --- | --- |
| 抬左前腳、停留、放回 | `horse-textured-wan-leftforeleg.yaml` | `horse-textured-wan-leftforeleg-normalized-rigid.yaml` |
| 輕抬尾巴、放回 | `horse-textured-wan-tail.yaml` | `horse-textured-wan-tail-normalized-rigid.yaml` |
| 張嘴、閉嘴 | `horse-textured-wan-mouth.yaml` | `horse-textured-wan-mouth-normalized-rigid.yaml` |

路徑均在 `configs/experiments/`。三組維持 `master` 的原始正向 prompt、1280×704
面積預算、121 幀、24 fps、seed 2026、50 steps、guidance scale 5，以及預訓練模型的
sample shift。25 個最佳化取樣幀不變。新關節模式的 `batch_frames` 為 1，
只降低後續關節渲染的尖峰用量，不影響參考影片。
每組原設定與新設定的 asset／camera／video 相同，都增加了
`cpu_export: true`，以獨立程序執行並在 CPU 編碼影片，沒有降低生成尺寸或幀數。
`master` 的舊 CLI 沒有傳入 YAML 的 negative prompt，而是使用模型預設負向提示；
兩份設定均明列 `use_negative_prompt: false`，保留相同的有效負向提示。
兩份設定共用同一份 video／supervision cache，避免比較不同生成結果。
原設定使用目前分支的預設 `paper` 關節模式，
不是 `master` 舊版關節程式，因此這是目前兩種模式的比較。

先試抬前腳，再試尾巴；目前馬的 skin joint 名稱未見獨立下顎關節，
張嘴影片即使生成成功，骨架也可能無法重現，較適合最後做能力邊界檢查。

```bash
# 先只生成影片，檢查指定動作、外形與鏡頭是否穩定
bash scripts/13_review_video.sh configs/experiments/horse-textured-wan-leftforeleg-normalized-rigid.yaml
# 影片合格後，用同一份設定繼續做新關節擬合
bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-leftforeleg-normalized-rigid.yaml --until-stage joints
# 同一段影片再跑原模式，作為比較
bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-leftforeleg.yaml --until-stage joints
```

尾巴、嘴巴將以上檔名中的 `leftforeleg` 換成 `tail`、`mouth`。
不使用 `master` 的 `*-e2e*.yaml`：那些設定指定另一環境的
`reference-framed.mp4` 路徑；這裡直接共用 pipeline cache，無需手動搬影片。

`cpu_export: true` 透過獨立程序呼叫官方 `WanTI2V.generate(max_area=...)`，
保留 CPU T5、CPU 初始化、模型 offload 與 bfloat16 權重，並將輸出移至 CPU 後再編碼。
預設使用官方 sample shift，不再根據解析度隱含更改。未指定尺寸且沒有啟用
`cpu_export` 的其他設定仍走原本的 704p CLI。不修改 `third_party`。
腳本預設啟用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，已設定的值會保留。
SAM2 的影格與推論狀態也都 offload 至 CPU，會增加 CPU RAM 用量。

先前 832×480／81 幀的降規設定另存為 `*-normalized-rigid-low-memory.yaml`，
只有明確選用才會降規；其中 sample shift 3 也明列於 YAML。主設定已恢復原始品質。
生成程式修改會改變 cache key；既有 cache 保留，可用 `reference_video` 重用成功影片。
目前只完成離線驗證，未在 24 GB GPU 實測峰值，其他程序占用 GPU 時仍可能 OOM。

先使用同一個資產、參考影片、prompt 與 seed 比較。論文核對與監督診斷見
[實作核對報告](paper-implementation-audit.md)。目前預設允許所有關節旋轉與位移，
額外穩定 prior 預設關閉。

像素中心轉換修正會使 render／supervision 的程式碼 cache key 改變；首次更新後
先用下列指令正常執行，讓快取逐階段檢查，不加 `--from-stage joints`。
若 canonical 圖片內容相同，既有 reference video 仍可命中快取，不需刪除或重生影片。

## 先跑到 joint，不執行 FEM

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/horse-textured-ltx-5s.yaml --until-stage joints
```

如果原本影片是用 Wan 生成，維持相同 backend：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/horse-textured-ltx-5s.yaml --video-backend wan --until-stage joints
```

已完成且相符的前置階段會命中快取。若希望前置階段必須已經有快取，
可再加 `--from-stage joints`；設定不一致或缺少快取時，程式會停止並指出階段。
不要刪除 `.cache/animimic`。

輸出在終端機印出的 `runs/<asset>/<run-id>/joint-review/`：

- `joint_0.mp4`：左參考影片、右 joint 結果。
- `joint_-45.mp4`、`joint_45.mp4`：左右斜側面。
- `joint.glb`：骨架動畫。
- `joint-review.json`：最佳化取樣數、輸出幀數與 fps。

MP4 與 GLB 按參考影片的實際 fps 輸出。以 121 幀、24 fps 的影片為例，
即使最佳化只用 25 個取樣，joint 預覽仍有 121 幀。中間姿勢由關節旋轉的
SLERP 與位移／尺度的線性插值計算，再做 FK/LBS；不直接線性混合兩個 mesh。
參考畫面讀取原影片對應影格。這改善低幀率顯示，不代表最佳化的姿勢抖動已解決。

## 非論文的穩定化實驗（預設關閉）

以下三項為本分支先前加入的實驗，不是論文原有約束。現在不會預設啟用。

1. 可選擇只允許根節點位移，子關節保持原本的 local offset，避免逐幀改變骨長。
2. 新增關節旋轉、根節點世界位置的時間加速度損失，使用取樣影格的實際秒數，
   讓相同動作在不同取樣幀數下有一致的尺度。
3. 對可信、整段影片幾乎不動的追蹤點，約束其 3D 表面位置保持穩定。
   判斷需要第一幀可信、至少 80% 影格可信（visibility × confidence > 0.5），
   且可信影格相對第一幀的位移都不超過預設 2 pixels。
   有明顯抬腳或移動的點不加入這項約束；它不依賴手動標記四蹄。

這是追蹤訊號驅動的穩定約束。遮擋、錯誤追蹤或錯誤深度仍需要檢查，
不應只靠 loss 下降判定成功。側面預覽尤其要確認隱藏的腿沒有異常轉動。

只有明確使用以下設定才會啟用這組實驗：

```yaml
joints:
  optimize_joint_translations: false
  stationary_track_px: 2.0
  loss_weights:
    rotation_smooth: 0.01
    root_smooth: 0.1
    stationary: 1.0
```

其他損失沿用既有設定。模型座標已正規化，靜止點損失以 0.01 的座標尺度
計算 Huber loss，減少較差觀測的極端影響。

目前預設是 `optimize_joint_translations: true`，新增三項權重皆為 0。
沿用沒有這些設定的既有 YAML 即可；若曾手動加上實驗設定，請刪除或改回上述預設。
主要 loss 與排程已按論文公式修改；N、bounds、norm 等未公開的選擇，詳見實作核對報告。

## 驗證

`test_joint_stability.py` 以靜止腳部、正常頭部動作加上逐幀深度雜訊建立測試：
檢查子關節 offset 不變、腳部時間二階差分降低，並保留正常頭部動作。
另外驗證移動／低信心點不被判成靜止點，以及不同取樣率的加速度計算。

`test_joint_review.py` 驗證旋轉插值保持長度，並實際寫出 MP4，檢查使用來源影片
的 fps／影格數，以及 `--until-stage joints` 不執行 tetrahedralization／FEM。
