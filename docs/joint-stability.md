# Joint 抖動修正與檢查

先使用同一個資產、參考影片、prompt 與 seed 比較。這次修改不改 video 或
supervision 的 cache key；關節損失與參數化改變後，只需要重新最佳化 joint。

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
參考畫面讀取原影片對應影格。這改善低幀率顯示，姿勢抖動由以下約束處理。

## 最佳化行為

1. 預設只允許根節點位移，子關節保持原本的 local offset，避免逐幀改變骨長。
2. 新增關節旋轉、根節點世界位置的時間加速度損失，使用取樣影格的實際秒數，
   讓相同動作在不同取樣幀數下有一致的尺度。
3. 對可信、整段影片幾乎不動的追蹤點，約束其 3D 表面位置保持穩定。
   判斷需要第一幀可信、至少 80% 影格可信（visibility × confidence > 0.5），
   且可信影格相對第一幀的位移都不超過預設 2 pixels。
   有明顯抬腳或移動的點不加入這項約束；它不依賴手動標記四蹄。

這是追蹤訊號驅動的穩定約束。遮擋、錯誤追蹤或錯誤深度仍需要檢查，
不應只靠 loss 下降判定成功。側面預覽尤其要確認隱藏的腿沒有異常轉動。

目前預設等同：

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

比較舊參數化時，可設 `optimize_joint_translations: true`，並將新增三項權重設為 0。
這不影響預覽的原生幀率輸出。

## 驗證

`test_joint_stability.py` 以靜止腳部、正常頭部動作加上逐幀深度雜訊建立測試：
檢查子關節 offset 不變、腳部時間二階差分降低，並保留正常頭部動作。
另外驗證移動／低信心點不被判成靜止點，以及不同取樣率的加速度計算。

`test_joint_review.py` 驗證旋轉插值保持長度，並實際寫出 MP4，檢查使用來源影片
的 fps／影格數，以及 `--until-stage joints` 不執行 tetrahedralization／FEM。
