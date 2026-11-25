# Joint 預覽與實驗設定

## 按骨架部位取點與檢查追蹤

抬前腳實驗可先跑到 supervision，在 GPU 主機執行

```bash
bash scripts/08_run_pipeline.sh configs/experiments/horse-textured-wan-leftforeleg-grouped-tracks.yaml --until-stage supervision --debug-images
```

新設定使用同一份影片設定與 512 點預算，依可見表面的蒙皮權重分配給各部位。
同一部位內由中央開始，反覆選離已選點最遠的可見像素。取不到點的部位會留空，
其額度分給其他可見部位。設定中的骨架索引適用於目前的 `horse_textured` GLB。

終端機會印出 `supervision-review.mp4` 的位置。追蹤點、SAM2 遮罩和 VGGT 深度會並排，
畫面上會標出來源影格。顏色代表點最初對應的部位。實心表示 visibility × confidence > 0.5
且位置落在 SAM2 遮罩內。這只用來篩查，仍要目視確認點是否跟在正確的腳上。
`--debug-images` 預設關閉，開啟後可以直接從 supervision 快取產圖。
`track-parts.csv` 記錄各部位在每幀的點數。看完後用同一份設定加
`--from-stage joints --until-stage joints` 重用追蹤資料進行骨架擬合。

## 關節擬合設定

未指定 `joints.objective` 時使用 `paper`。馬的 grouped-tracks 設定使用
`normalized`，會正規化 RGB、輪廓和追蹤 loss，依追蹤信心篩點，並在每步只渲染
`joints.batch_frames` 指定的幀數。Tracking、depth 和頂點平滑仍使用全部取樣幀。
`joints.loss_weights.rigid` 控制 Stage 1 的邊長約束，設為 0 時關閉。

`joint-review/joint_diagnostics.json` 記錄邊長變形與頂點加速度。
修改 joint 設定後，可用快取沿用相同影片與 supervision。
需要測不同動作時，複製 grouped-tracks 設定，在副本改 `video.prompt`，
或設定 `reference_video` 指向現成影片。

## 先跑到 joint，不執行 FEM

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/08_run_pipeline.sh configs/horse.yaml --until-stage joints
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
