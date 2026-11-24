# AniMimic 重現：用 Wan2.2 測試馬的細部動作

2026-11-19 · repo commit `93c9093` · 實驗主機：RTX A6000

## 摘要

- 影片生成模型從 **LTX-Video 2B 換成 Wan2.2 TI2V-5B**。LTX 做不出細部動作，Wan 三個動作都做得出來。
- 三個細部動作都**跑完 8 個 stage**：張嘴、抬左前腳、舉尾巴。

| 動作 | Wan 影片 | 骨架擬合（3.1） | 骨架 + FEM（3.2） | mask IoU（靜止 → 骨架 → +FEM） |
|---|---|---|---|---|
| 舉尾巴 | ✅ | ✅ 舉起和放下都有跟上 | ✅ | 0.903 → **0.949** → 0.927 |
| 張嘴 | ✅ | ⚠️ 只做出抬頭；骨架沒有下顎骨 | ⚠️ | 0.893 → **0.955** → 0.935 |
| 抬左前腳 | ✅ | ❌ 腳沒抬起來 | ❌ | 0.848 → **0.905** → 0.872 |

- 骨架擬合在三個動作上都有提升 IoU，**FEM 則讓 IoU 下降約 0.02–0.03**。目前的材料參數設定還沒帶來改善，見[已知問題](#已知問題與下一步)。
- FEM 原本跑不完，改寫成稀疏 Newton 求解器後，200 次迭代的材料優化可以穩定跑完。

---

## Pipeline

```
3D 馬（帶骨架的 GLB）
 → ① prepare      讀網格、貼圖、骨架（41 個關節）
 → ② render       用固定相機渲染第一幀
 → ③ video        Wan2.2：第一幀 + 文字 prompt → 5 秒影片（121 幀、24fps）
 → ④ supervision  取 25 幀：SAM2 輪廓、AllTracker 追蹤點、VGGT 深度
 → ⑤ joints       優化每幀關節旋轉，讓 LBS 變形後的網格對上影片      ← 論文 3.1
 → ⑥ tetrahedralize  fTetWild 把網格填成約 1.2 萬個四面體
 → ⑦ physics      骨頭帶動固定節點，FEM 模擬軟組織，優化楊氏模數 E   ← 論文 3.2
 → ⑧ export       joint.glb、預覽影片、IoU 和 track error
```

跟論文的差異：
- 論文的影片模型是 Kling 2.1，這裡是 Wan2.2 TI2V-5B。
- 論文用 UniRig 自動綁骨架，這裡直接用模型自帶的骨架。

---

## 結果

每張比較圖由左到右是：Wan 參考影片 | 骨架擬合 | 骨架 + FEM，由上到下取 5 個時間點。

### 舉尾巴（三個裡面最成功）

> The brown horse stands still and slightly raises its tail … then lets the tail settle back down.

![tail](runs/horse_textured/cb7d7555fb0d/final/comparison.jpg)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.903 | **0.949** | 0.927 |
| track error (px) ↓ | **1.49** | 4.29 | 3.96 |

- 尾巴有 4 節骨頭，能彎成曲線，所以舉起和放下都跟得上。
- 這是唯一一個 FEM 讓 track error 變好的動作（4.29 → 3.96）。
- 影片：[joint_0](runs/horse_textured/cb7d7555fb0d/final/joint_0.mp4) · [physics_0](runs/horse_textured/cb7d7555fb0d/final/physics_0.mp4) · [Wan 原片](runs/horse_textured/wan-tail/reference-framed.mp4) · 3D 檔 [joint.glb](runs/horse_textured/cb7d7555fb0d/final/joint.glb)

### 張嘴

![mouth](runs/horse_textured/0df6ed7ea361/final/comparison.jpg)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.893 | **0.955** | 0.935 |
| track error (px) ↓ | 4.60 | **4.03** | 4.21 |

- Wan 生成的影片裡，馬張嘴時頭也一起抬起來。
- 骨架擬合做出了**抬頭**，但做不出**張嘴**：頭部只有一個 Head 關節，控制 835 個頂點，沒有下顎骨，見[附錄骨架圖](#附錄-b馬的骨架)。
- FEM 讓頭部前端跟不上骨架，抬頭的幅度變小。差距最大的一幀約為馬身長的 11%。
- 影片：[joint_0](runs/horse_textured/0df6ed7ea361/final/joint_0.mp4) · [physics_0](runs/horse_textured/0df6ed7ea361/final/physics_0.mp4) · [Wan 原片](runs/horse_textured/wan-mouth/reference-framed.mp4)

### 抬左前腳

![leftforeleg](runs/horse_textured/072efd149a3d/final/comparison.jpg)

| | 靜止 | 骨架擬合 | 骨架 + FEM |
|---|---|---|---|
| mask IoU ↑ | 0.848 | **0.905** | 0.872 |
| track error (px) ↓ | **2.67** | 4.99 | 4.97 |

- Wan 影片裡有抬腳，另外還多甩了一下尾巴。骨架擬合只跟上了尾巴，**四條腿都是直的**。
- 前腳有 UpperArm、Forearm、Hand 三節骨頭，所以不是骨架的限制，比較可能是優化卡在局部最小值：抬起的腳和身體在畫面上重疊，輪廓 loss 推不動它。
- 完整版 FEM 在第 42 次迭代失敗，因為有四面體被壓到翻轉。這裡放的是只跑 15+10 次迭代的快速版。
- 影片：[joint_0](runs/horse_textured/072efd149a3d/final/joint_0.mp4) · [physics_0](runs/horse_textured/072efd149a3d/final/physics_0.mp4) · [Wan 原片](runs/horse_textured/wan-leftforeleg/reference-framed.mp4)

---

## 為什麼從 LTX 換成 Wan

同一張輸入圖、同樣三個 prompt，LTX-Video 2B 做不出任何一個細部動作，Wan2.2 三個都做得出來：

![wan vs ltx](runs/horse_textured/wan-vs-ltx-fine-actions.jpg)

LTX 在更早的測試裡也有同樣的狀況（[紀錄](docs/ltx-experiments.md)）：
- 只有 5 秒內的小動作（低頭）可以用。
- 20 秒影片不是頭頸融化，就是整段靜止。
- 翻身、轉身這類大動作，不是不動就是結構崩壞。

接上 Wan 時，有兩個地方需要處理：
- **畫面裁切**：Wan 會把輸入圖縮放後中央裁切，畫面就跟 3D 相機對不上。生成後把裁掉的邊補回來，第一幀與輸入圖的差異從 4.44 降到 1.69。
- **生成參數**：官方的 `generate.py` 不能設 negative prompt，所以改成直接呼叫 `WanTI2V` API。

---

## FEM 工程

原本的 FEM 實作跑不完：光是初始姿勢就跑了 45 分鐘以上，還沒收斂。

主要的修改：
1. 把 CG 迭代換成**稀疏直接求解**（SuperLU），並快取稀疏結構。
2. 推導 fixed-corotated 模型的**解析 Hessian**，再用 polar decomposition 加 eigh 取代 GPU 上的批次 SVD。Hessian 的計算時間從 192 ms 降到 49 ms。
3. **Warm start**：每次迭代都從上一次迭代同一步的解開始算。

速度的變化：

| 版本 | 每次迭代 |
|---|---|
| 原本 | 跑不完 |
| 第一階段 | 20 分鐘 |
| 第二階段 | 7.3 分鐘 |
| 最終版 | 約 47 秒 |

200 次迭代單獨跑約 2 小時 40 分。這次三個動作同時跑，又和別人共用 GPU，每個約 4 小時。

這次另外加了**失敗保護**：某次迭代模擬失敗時，退回上一組成功的材料參數，並把學習率減半，不再讓整個 pipeline 中止。

---

## 已知問題與下一步

| 問題 | 可能原因 | 下一步 |
|---|---|---|
| FEM 都讓 IoU 下降 | 初始 E = 1e4 太軟，learning rate 1e-3 太小，200 次迭代 E 只從 1e4 變到 0.8–1.2 萬 | 調初始 E 和 learning rate |
| 抬左前腳沒做出來 | 骨架擬合卡在局部最小值 | 檢查 track loss 的權重、增加關節旋轉的初始化 |
| 骨架擬合後 track error 比靜止還差（抬腳、舉尾巴） | track loss 幾乎是 0，沒發揮作用；加入 rgb loss 後 mask loss 也變差 | 檢查 AllTracker 軌跡和 loss 權重 |
| 張嘴做不出來 | 骨架沒有下顎骨 | 換有臉部骨架的模型，或用 UniRig 重新綁骨架 |
| FEM 偶爾壓壞四面體 | fTetWild 每次產生的網格都不同，有些網格比較容易壞 | 固定網格、限制 E 的下限 |

---

## 附錄 A：所有實驗影片（依時間順序）

> `runs/` 只存在實驗主機上，不在 git 裡；`docs/` 底下的已經 commit。

**1. LTX-Video 2B 影片生成（11/18 22:08–22:24）**：只生成影片，詳見 [docs/ltx-experiments.md](docs/ltx-experiments.md)

| 實驗 | 影片 | 結果 |
|---|---|---|
| 5 秒基準 | [5s](docs/ltx-experiments/5s/reference.mp4) | ✅ 只有低頭 |
| 20 秒（481 幀、24fps） | [20s-A](docs/ltx-experiments/20s-A-481f-24fps/reference.mp4) | ❌ 頭頸融化 |
| 20 秒（241 幀、12fps） | [20s-B](docs/ltx-experiments/20s-B-241f-12fps/reference.mp4) | ❌ 靜止 |
| 翻身 ×3 | [roll-s2026](docs/ltx-experiments/roll-s2026/reference.mp4) · [roll-s7](docs/ltx-experiments/roll-s7/reference.mp4) · [roll2-s2026](docs/ltx-experiments/roll2-s2026/reference.mp4) | ❌ |
| 轉身 ×2 | [turn-s2026](docs/ltx-experiments/turn-s2026/reference.mp4) · [turn-s7](docs/ltx-experiments/turn-s7/reference.mp4) | ❌ |

**2. 第一次完整 pipeline：LTX 5 秒低頭（11/18 23:43 → 11/19 04:48）**

- 骨架擬合：[joint_views.mp4](runs/horse_textured/46dd1bff7e3c/final/joint_views.mp4)（2×2：左上參考影片、右上 0°、左下 −45°、右下 +45°）
- 骨架 + FEM：[physics_views.mp4](runs/horse_textured/46dd1bff7e3c/final/physics_views.mp4)
- LTX 原片：[reference.mp4](docs/ltx-experiments/5s/reference.mp4) · 3D 檔：[joint.glb](runs/horse_textured/46dd1bff7e3c/final/joint.glb)
- 比較圖：[comparison.jpg](runs/horse_textured/46dd1bff7e3c/final/comparison.jpg)
- mask IoU 0.884 → **0.964** → 0.947；track error 6.29 → **2.82** → 3.34 px

**3. LTX 細部動作（11/19 12:13）**

- 影片：[張嘴](runs/horse_textured/ltx-mouth/reference.mp4) · [抬左前腳](runs/horse_textured/ltx-leftforeleg/reference.mp4) · [舉尾巴](runs/horse_textured/ltx-tail/reference.mp4)
- 比較圖：[fine-actions-compare.jpg](runs/horse_textured/fine-actions-compare.jpg) · [crops](runs/horse_textured/fine-actions-crops.jpg)

**4. Wan2.2 細部動作（11/19 12:52）**

121 幀、24fps、50 steps、guidance 5.0、seed 2026。

| 動作 | 原始輸出 1216×736 | 補邊後 1262×736（pipeline 使用） |
|---|---|---|
| 張嘴 | [reference](runs/horse_textured/wan-mouth/reference.mp4) | [reference-framed](runs/horse_textured/wan-mouth/reference-framed.mp4) |
| 抬左前腳 | [reference](runs/horse_textured/wan-leftforeleg/reference.mp4) | [reference-framed](runs/horse_textured/wan-leftforeleg/reference-framed.mp4) |
| 舉尾巴 | [reference](runs/horse_textured/wan-tail/reference.mp4) | [reference-framed](runs/horse_textured/wan-tail/reference-framed.mp4) |

**5. Wan 細部動作的完整 pipeline（11/19 12:55 → 17:58）**

每支影片是 2×2 排列，左上是參考影片（Wan），右上是原相機 0°，左下是相機轉 −45°，右下是相機轉 +45°，每格上方有標籤。
影片是從 3D 結果以 2 倍解析度（每格 1536×896）重新渲染的，參考影片取自 Wan 原片的對應幀。

| 動作 | 骨架擬合 | 骨架 + FEM | 其他 |
|---|---|---|---|
| 張嘴 | [joint_views.mp4](runs/horse_textured/0df6ed7ea361/final/joint_views.mp4) | [physics_views.mp4](runs/horse_textured/0df6ed7ea361/final/physics_views.mp4) | [glb](runs/horse_textured/0df6ed7ea361/final/joint.glb) · [metrics](runs/horse_textured/0df6ed7ea361/final/metrics.json) |
| 抬左前腳 | [joint_views.mp4](runs/horse_textured/072efd149a3d/final/joint_views.mp4) | [physics_views.mp4](runs/horse_textured/072efd149a3d/final/physics_views.mp4) | [glb](runs/horse_textured/072efd149a3d/final/joint.glb) · [metrics](runs/horse_textured/072efd149a3d/final/metrics.json) |
| 舉尾巴 | [joint_views.mp4](runs/horse_textured/cb7d7555fb0d/final/joint_views.mp4) | [physics_views.mp4](runs/horse_textured/cb7d7555fb0d/final/physics_views.mp4) | [glb](runs/horse_textured/cb7d7555fb0d/final/joint.glb) · [metrics](runs/horse_textured/cb7d7555fb0d/final/metrics.json) |

原始的單一角度影片（`joint_0/45/-45.mp4`、`physics_0/45/-45.mp4`）仍在同一個 `final/` 資料夾裡。

三個動作的骨架擬合對照（不含 FEM）：
- 總覽圖：[all_reference_vs_joint.jpg](runs/horse_textured/joint-previews/all_reference_vs_joint.jpg)
- 影片：[張嘴](runs/horse_textured/joint-previews/mouth_reference_vs_joint.mp4) · [抬左前腳](runs/horse_textured/joint-previews/leftforeleg_reference_vs_joint.mp4) · [舉尾巴](runs/horse_textured/joint-previews/tail_reference_vs_joint.mp4)

## 附錄 B：馬的骨架

模型：Sketchfab "Horse Rigged (Game Ready)"，作者 abhayexe，授權 CC BY 4.0。glTF 2.0 格式，1,903 個頂點、2,914 個三角形。

![rig](runs/horse_textured/rig-preview/rig.png)

GLB 裡共有 41 個 `Bip01` 關節，其中 31 個有綁定網格頂點，完整清單在 [joints.txt](runs/horse_textured/rig-preview/joints.txt)：

| 部位 | 關節 |
|---|---|
| 軀幹 | Pelvis → Spine → Spine1 |
| 頸、頭 | Neck → Neck1 → Head（一個關節控制整顆頭，沒有下顎骨） |
| 耳朵 | Ponytail1、Ponytail2 |
| 前腳 ×2 | Clavicle → UpperArm → Forearm → Hand → Finger0 |
| 後腳 ×2 | Thigh → Calf → HorseLink → Foot → Toe0 |
| 尾巴 | Tail → Tail1 → Tail2 → Tail3 |
