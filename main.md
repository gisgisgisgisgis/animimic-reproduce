# 實驗影片總覽（依實驗順序）

模型：`horse_textured`（42 個關節的綁骨馬），固定相機（方位角 65°、仰角 5°）、768×448。
輸入圖：[docs/asset-previews/horse-textured-input.png](docs/asset-previews/horse-textured-input.png)

> `runs/` 底下的檔案不在 git 裡，只存在實驗主機上；`docs/` 底下的已經 commit。

---

## 1. LTX-Video 2B 影片生成（11/18 22:08–22:24）

只跑影片生成，還沒接後面的 pipeline。詳細紀錄見 [docs/ltx-experiments.md](docs/ltx-experiments.md)。

| # | 實驗 | 影片 | 結果 |
|---|---|---|---|
| 1-1 | 基準 5 秒（121 幀、24fps） | [5s](docs/ltx-experiments/5s/reference.mp4) | ✅ 只做出低頭，最後 1 秒臉部變形 |
| 1-2 | 20 秒 A（481 幀、24fps） | [20s-A](docs/ltx-experiments/20s-A-481f-24fps/reference.mp4) | ❌ 約 3 秒後頭頸融化，之後靜止 |
| 1-3 | 20 秒 B（241 幀、12fps） | [20s-B](docs/ltx-experiments/20s-B-241f-12fps/reference.mp4) | ❌ 幾乎是靜止圖片 |
| 1-4 | 翻身 seed 2026 | [roll-s2026](docs/ltx-experiments/roll-s2026/reference.mp4) | ❌ 低頭彎腿，沒有躺下 |
| 1-5 | 翻身 seed 7 | [roll-s7](docs/ltx-experiments/roll-s7/reference.mp4) | ⚠️ 趴倒但沒翻過來，嚴重變形 |
| 1-6 | 跪下再翻身 | [roll2-s2026](docs/ltx-experiments/roll2-s2026/reference.mp4) | ❌ 幾乎不動 |
| 1-7 | 原地轉身 seed 2026 | [turn-s2026](docs/ltx-experiments/turn-s2026/reference.mp4) | ❌ 幾乎不動，頭部拉長 |
| 1-8 | 原地轉身 seed 7 | [turn-s7](docs/ltx-experiments/turn-s7/reference.mp4) | ❌ 全程靜止 |

比較圖：[roll-turn-compare.jpg](docs/ltx-experiments/roll-turn-compare.jpg)

**結論**：LTX 2B 只能做出 5 秒內的小動作，長片和大動作都會失敗。

---

## 2. 第一次完整 pipeline：LTX 5 秒低頭（11/18 23:43 → 11/19 04:48）

用 1-1 的影片跑完整 8 個 stage。FEM 是改成稀疏 Newton 求解器之後，第一次完整跑完（200 次迭代，約 2 小時 39 分）。

| 階段 | 影片 |
|---|---|
| 骨架擬合（論文 3.1），FEM 前先看的版本 | [joint_0](runs/horse_textured/46dd1bff7e3c/joint-only/joint_0.mp4) · [joint_-45](runs/horse_textured/46dd1bff7e3c/joint-only/joint_-45.mp4) · [joint_45](runs/horse_textured/46dd1bff7e3c/joint-only/joint_45.mp4) |
| 骨架擬合（最終輸出） | [joint_0](runs/horse_textured/46dd1bff7e3c/final/joint_0.mp4) · [joint_-45](runs/horse_textured/46dd1bff7e3c/final/joint_-45.mp4) · [joint_45](runs/horse_textured/46dd1bff7e3c/final/joint_45.mp4) |
| 骨架 + FEM（論文 3.2） | [physics_0](runs/horse_textured/46dd1bff7e3c/final/physics_0.mp4) · [physics_-45](runs/horse_textured/46dd1bff7e3c/final/physics_-45.mp4) · [physics_45](runs/horse_textured/46dd1bff7e3c/final/physics_45.mp4) |

`_0` 是原相機角度，左邊並排參考影片；`_-45` 和 `_45` 是左右各轉 45° 的視角。
比較圖：[comparison.jpg](runs/horse_textured/46dd1bff7e3c/final/comparison.jpg) · 3D 檔：[joint.glb](runs/horse_textured/46dd1bff7e3c/final/joint.glb)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.884 | **0.964** | 0.947 |
| track error (px) ↓ | 6.29 | **2.82** | 3.34 |

---

## 3. 細部動作測試：LTX（11/19 12:13）

三個 prompt：張開嘴巴、抬左前腳、尾巴稍微舉起來。

| 動作 | 影片 |
|---|---|
| 張嘴 | [ltx-mouth](runs/horse_textured/ltx-mouth/reference.mp4) |
| 抬左前腳 | [ltx-leftforeleg](runs/horse_textured/ltx-leftforeleg/reference.mp4) |
| 舉尾巴 | [ltx-tail](runs/horse_textured/ltx-tail/reference.mp4) |

比較圖：[fine-actions-compare.jpg](runs/horse_textured/fine-actions-compare.jpg) · 局部放大：[fine-actions-crops.jpg](runs/horse_textured/fine-actions-crops.jpg)

**結論**：LTX 2B 做不出這些細部動作，因此改用 Wan。

---

## 4. 細部動作測試：Wan2.2 TI2V-5B（11/19 12:52）

設定 121 幀、24fps、50 steps、guidance 5.0、seed 2026。

| 動作 | Wan 原始輸出（1216×736） | 補回被裁掉的邊（1262×736，pipeline 實際使用） | 觀察 |
|---|---|---|---|
| 張嘴 | [reference](runs/horse_textured/wan-mouth/reference.mp4) | [reference-framed](runs/horse_textured/wan-mouth/reference-framed.mp4) | ✅ 嘴張開，頭也跟著抬起 |
| 抬左前腳 | [reference](runs/horse_textured/wan-leftforeleg/reference.mp4) | [reference-framed](runs/horse_textured/wan-leftforeleg/reference-framed.mp4) | ✅ 前腳抬起，另外多甩了一下尾巴 |
| 舉尾巴 | [reference](runs/horse_textured/wan-tail/reference.mp4) | [reference-framed](runs/horse_textured/wan-tail/reference-framed.mp4) | ⚠️ 尾巴舉得比要求高，尾毛變得比較蓬 |

Wan 和 LTX 的比較圖：[wan-vs-ltx-fine-actions.jpg](runs/horse_textured/wan-vs-ltx-fine-actions.jpg)

---

## 5. Wan 細部動作跑完整 pipeline（11/19 12:55 起）

### 5-1. 骨架擬合結果（三個都完成）

每支影片左邊是 Wan 參考影片，右邊是骨架擬合（論文 3.1）。

| 動作 | 影片 | 結果 |
|---|---|---|
| 張嘴 | [mouth_reference_vs_joint](runs/horse_textured/joint-previews/mouth_reference_vs_joint.mp4) | 抬頭有做出來；張嘴做不出來（骨架沒有下顎關節） |
| 抬左前腳 | [leftforeleg_reference_vs_joint](runs/horse_textured/joint-previews/leftforeleg_reference_vs_joint.mp4) | 尾巴有跟上，**抬腳沒有做出來** |
| 舉尾巴 | [tail_reference_vs_joint](runs/horse_textured/joint-previews/tail_reference_vs_joint.mp4) | ✅ 跟得很好 |

總覽圖：[all_reference_vs_joint.jpg](runs/horse_textured/joint-previews/all_reference_vs_joint.jpg)

### 5-2. 張嘴：完整結果（骨架 + FEM，17:21 完成）

| 階段 | 影片 |
|---|---|
| 骨架擬合 | [joint_0](runs/horse_textured/0df6ed7ea361/final/joint_0.mp4) · [joint_-45](runs/horse_textured/0df6ed7ea361/final/joint_-45.mp4) · [joint_45](runs/horse_textured/0df6ed7ea361/final/joint_45.mp4) |
| 骨架 + FEM | [physics_0](runs/horse_textured/0df6ed7ea361/final/physics_0.mp4) · [physics_-45](runs/horse_textured/0df6ed7ea361/final/physics_-45.mp4) · [physics_45](runs/horse_textured/0df6ed7ea361/final/physics_45.mp4) |

比較圖：[comparison.jpg](runs/horse_textured/0df6ed7ea361/final/comparison.jpg) · 3D 檔：[joint.glb](runs/horse_textured/0df6ed7ea361/final/joint.glb)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.893 | **0.955** | 0.935 |
| track error (px) ↓ | 4.60 | **4.03** | 4.21 |

FEM 讓抬頭的幅度變小，數值也比只做骨架擬合差一點，跟第 2 節的結果一致。

### 5-3. 抬左前腳：完整結果（快速版 FEM，17:37 完成）

完整版 FEM（100+100 次迭代）在 13:42 失敗：第 42 次迭代時，第 18 幀有四面體被壓到翻轉。
這裡是快速版，FEM 只跑 15+10 次迭代。

| 階段 | 影片 |
|---|---|
| 骨架擬合 | [joint_0](runs/horse_textured/072efd149a3d/final/joint_0.mp4) · [joint_-45](runs/horse_textured/072efd149a3d/final/joint_-45.mp4) · [joint_45](runs/horse_textured/072efd149a3d/final/joint_45.mp4) |
| 骨架 + FEM | [physics_0](runs/horse_textured/072efd149a3d/final/physics_0.mp4) · [physics_-45](runs/horse_textured/072efd149a3d/final/physics_-45.mp4) · [physics_45](runs/horse_textured/072efd149a3d/final/physics_45.mp4) |

比較圖：[comparison.jpg](runs/horse_textured/072efd149a3d/final/comparison.jpg) · 3D 檔：[joint.glb](runs/horse_textured/072efd149a3d/final/joint.glb)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.848 | **0.905** | 0.872 |
| track error (px) ↓ | **2.67** | 4.99 | 4.97 |

FEM loss 從 0.6026 降到 0.6017。抬腳在骨架擬合那一步就沒做出來，所以 IoU 是三個動作裡最低的。
track error 反而比靜止還差，代表追蹤點的對齊沒有帶來幫助。

### 5-4. 舉尾巴：完整結果（完整版 FEM 100+100 次，17:58 完成）

| 階段 | 影片 |
|---|---|
| 骨架擬合 | [joint_0](runs/horse_textured/cb7d7555fb0d/final/joint_0.mp4) · [joint_-45](runs/horse_textured/cb7d7555fb0d/final/joint_-45.mp4) · [joint_45](runs/horse_textured/cb7d7555fb0d/final/joint_45.mp4) |
| 骨架 + FEM | [physics_0](runs/horse_textured/cb7d7555fb0d/final/physics_0.mp4) · [physics_-45](runs/horse_textured/cb7d7555fb0d/final/physics_-45.mp4) · [physics_45](runs/horse_textured/cb7d7555fb0d/final/physics_45.mp4) |

比較圖：[comparison.jpg](runs/horse_textured/cb7d7555fb0d/final/comparison.jpg) · 3D 檔：[joint.glb](runs/horse_textured/cb7d7555fb0d/final/joint.glb)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.903 | **0.949** | 0.927 |
| track error (px) ↓ | **1.49** | 4.29 | 3.96 |

FEM loss 從 0.3859 降到 0.3807（coarse 結束時是 0.3830）。

- 舉尾巴和放下兩個動作，骨架擬合和 FEM 都有跟上，是三個動作裡最成功的。
- 這是唯一一個 FEM 讓 track error 變好的動作（4.29 → 3.96），不過 IoU 還是下降。
- 跟 5-3 一樣，track error 比靜止還差。追蹤點的監督還需要檢查。

快速版（10+5 次迭代）沒有跑成：第一次迭代的第 6 幀就有四面體被壓到翻轉。

---

## 總結：三個 Wan 細部動作

| 動作 | 影片做出來了嗎 | 骨架擬合 | FEM | mask IoU（靜止 → 骨架 → +FEM） |
|---|---|---|---|---|
| 張嘴 | ✅ | ⚠️ 只做出抬頭，沒有下顎骨所以不能張嘴 | 完整版 200 次 | 0.893 → 0.955 → 0.935 |
| 抬左前腳 | ✅ | ❌ 腳沒抬起來，只有尾巴跟上 | 快速版 25 次 | 0.848 → 0.905 → 0.872 |
| 舉尾巴 | ✅ | ✅ | 完整版 200 次 | 0.903 → 0.949 → 0.927 |

共同現象：
- 骨架擬合都有提升 IoU。
- FEM 都讓 IoU 下降約 0.02–0.03。可能是初始 E 1e4 太軟，加上 learning rate 1e-3 太小。

---

## 附錄：馬的骨架

- 骨架圖：[rig.png](runs/horse_textured/rig-preview/rig.png)。紅點是關節、黑線是骨頭，網格依受哪個關節控制最多上色。
- 關節清單：[joints.txt](runs/horse_textured/rig-preview/joints.txt)

GLB 裡共有 41 個 `Bip01` 關節，其中 31 個有綁定網格頂點：

| 部位 | 關節 |
|---|---|
| 軀幹 | Pelvis → Spine → Spine1 |
| 頸、頭 | Neck → Neck1 → Head |
| 耳朵 | Ponytail1、Ponytail2 |
| 前腳 ×2 | Clavicle → UpperArm → Forearm → Hand → Finger0 |
| 後腳 ×2 | Thigh → Calf → HorseLink → Foot → Toe0 |
| 尾巴 | Tail → Tail1 → Tail2 → Tail3 |

頭部只有一個 Head 關節，控制 835 個頂點，沒有下顎骨，所以張嘴做不出來。
