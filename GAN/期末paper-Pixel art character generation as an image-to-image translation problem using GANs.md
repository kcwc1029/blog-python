
## 1. Abstract

問題背景：Pixel art 中的角色設計需要大量反覆運算，修改一個角度會影響其他角度。
核心任務：Framed as an image-to-image translation task：
- e.g., generating a sprite facing right from one facing front.
方法架構：Build on Pix2Pix’s GAN architecture, leveraging characteristics of pixel art.
資料集：Used both small datasets (<1k) and a large one (12k).
貢獻：We hypothesize that the problem of generating character sprites in a target pose (e.g., facing right) given a source (e.g., facing front) can be framed as an image-to-image translation task
- 提出了一個 mixed-initiative sprite editor 讓人類與模型共創角色圖。
- 使用 GAN 處理「角色不同視角生成」問題。


![[image.png]]

## 2. 文獻探討：Image-to-Image Translation
### 2.1. 什麼是 image-to-image translation
> Pang et al. [11] define image-to-image translation as the process of converting an input image 𝑥𝐴 in a source domain 𝐴 to a target domain 𝐵 while keeping some intrinsic content from 𝐴 and transferring it to the extrinsic style of 𝐵

定義：把某張圖（來源領域A）轉成另一張圖（目標領域B）
- 同時保留原圖的內在內容（intrinsic content）
- 並應用新圖的外在風格（extrinsic style）




- 在這篇論文中，研究者將「角色不同視角（正面、側面、背面等）」視為不同的「domain（領域）」。
-

### 2.2. 內在內容 vs. 外在風格

> The intrinsic content of the source domain is the character phenotype features, accessories, and palette, while the extrinsic style of the target domain comprises the arrangement of pixels in the new pose.

> In this work, we frame the approached problem as an image-to-image translation task, with the domains representing different character poses: front, right, back, and left.

- 內在內容（intrinsic content）：角色的基本特徵
	- 如：髮型、衣服、配件、顏色（palette）
- 外在風格（extrinsic style）：角色的新姿勢下，像素如何排列

### 2.3. Pix2Pix

> An influential work on image-to-image translation is Pix2Pix

> It is a general architecture of deep generative models that proposes to condition the generation process on an image in the source domain while doing supervised training for the generator to create its version in a target domain.

- Pix2Pix 是影像轉影像（image-to-image translation）領域中的代表性模型。
- Pix2Pix 是一種條件式生成模型（conditional GAN）。
- 屬於有監督學習（supervised learning），因為訓練時有成對資料。
- 它的生成器（Generator）會根據輸入圖片（source domain）產生對應的輸出圖（target domain）。
	- Generator：使用 U-Net 結構，保留細節與圖像結構。輸入與輸出圖片解析度相同，但風格/內容不同。
	- Discriminator（鑑別器）：
		- 不只判斷整張圖真假，而是對圖中的每個小區塊（patch）做判別。
		- 有助於學習局部細節，尤其適合像 pixel art 這種每個像素都很重要的圖像。
- 應用： Pix2Pix 可應用於各種圖像對應任務
	- 白天照片 ↔ 夜間照片
	- 黑白圖 ↔ 彩色圖
	- 區域標記圖 ↔ 實景照片

### 2.4. 先前研究
> Translate line art sketches of characters into a grayscale and a colored sprite [...] 85 and 530 examples.”

- Serpa & Rodrigues (2019)利用 Pix2Pix 將角色線稿轉成灰階與彩色 sprite。
	- 成效：灰階圖效果好，彩色圖有很多高頻雜訊，特別是在沒看過的姿勢（unseen poses）
	- 問題：訓練資料太少，模型泛化能力差

> Adapted Pix2Pix [...] grayscale to colored images [...] YUV inputs yielded better results than RGB.

- Jiang & Sweetser (2021)利用 Pix2Pix將黑白圖渲染成彩色圖(這也是我們想要完成的期末專案)
	- 使用 YUV 色彩空間（而不是 RGB）
	- 因為只需要學 UV（色彩），Y（亮度）可視為條件參考值。

> Proposed a CVAE to transfer the domain of a Pokémon [...] blurry results with noisy colors

- Gonzalez et al. (2020)使用 CVAE（變分自編碼器）將 Pokémon 圖從一種類型（如火）轉為另一種（如草）。
	- 問題01：結果模糊、不精確，顏色亂、邊緣不清
	- 問題02：原因之一是資料量太小（974 張圖）
> Proposed a multiple discriminators GAN ... trained with bone-graph pose image and a color-shape image

-  Hong et al. (2019) 使用兩個鑑別器的 GAN（MDGAN）
	- 一個判斷「形狀+顏色是否正確」
	- 一個判斷「姿勢是否與骨架圖匹配」
	- 限制01：需要額外的骨架圖（bone-graph），資料製作成本高
	- 限制02：僅適用「相同形狀角色」→ 限制創作多樣性

### 2.5. 本研究的差異與貢獻（與上述相比）

> Our work also adapts Pix2Pix [...] But differently from the aforementioned works [...]

| 比較項目      | 本研究的優勢                   |
| --------- | ------------------------ |
| 是否限制角色形狀  | ❌ 不限制，支援不同角色樣貌           |
| 是否需要骨架資料集 | ❌ 不需要人工骨架（與 MDGAN 不同）    |
| 色彩空間選擇    | ✅ 使用 RGBA（比 RGB 更保留透明資訊） |
| 任務目標      | ✅ 專注於不同「視角」的轉換（正面 → 側面）  |
| 模型泛化能力    | ✅ 探討多樣資料集如何影響生成品質        |



## 3. 文獻探討：Generating pixel art


![[image-2.png]]

### 3.1. 解析度低 → 每個像素都超重要

>Pixel art images frequently have a lower resolution, so each pixel encodes more information.

> A one-pixel difference in a character’s face can change its expression　

 - pixel art 的解析度很低（通常是 32x32 或 64x64），
	- 每一個像素都代表重要的視覺資訊！
	- 一個像素改變 → 就可能讓角色表情完全不同。

### 3.2. 頻率分布與照片不同

> The distribution of frequencies along the image is also different [...] low-frequency regions interleave more quickly with high-frequency parts

- 相比照片，pixel art顏色區塊（低頻）與邊緣變化（高頻）交錯得更密集
	- 圖片小 → 色塊與邊界更近

### 3.3. 調色限制帶來獨特技巧
> Pixel artists compose images by selecting colors from a small palette, frequently with less than thirty colors

- pixel art 常只用 30 種顏色以下，因此發展出一些專門技術：
	- Color ramps：用少量顏色表現漸層效果
	- Dithering：把兩種顏色交錯排列來混色，形成視覺混合

### 3.4. pixel art 資料集比較少
> Pixel art datasets are much more scarce than photography-based ones.
-  與真實照片相比，pixel art 的訓練資料集很稀少、很難找。

## 4. 文獻探討：Mixed-initiative content generation（混合主動內容生成）

### 4.1. 什麼是 Mixed-Initiative
> Some procedural content generation methods are fully automated, others involve more human interaction [...] automate only part of the process

> Task initiative, speaker initiative, outcome initiative

> An iterative loop where at least one agent [...] can take the initiative [...] all involved agents [...] can contribute [...] and respond at least once

- 人與 AI一起內容生成（像是遊戲角色、關卡、美術素材等）
- Liapis et al.（2016）定義的三種「主動性」
	- 在這種系統中，人和 AI 都可以「主動」做事，包括
	- Task initiative：誰定義工作要做什麼？
	- Speaker initiative：誰決定何時開始做？
	- Outcome initiative：誰來決定怎麼做最好？

- Lai et al.（2022）對 Mixed-Initiative 的嚴格定義
	- 至少有一個參與者可以「主動發起」創作
	- 人與 AI 都要能「對內容做出貢獻」
	-  每個參與者都要「至少互動一次」


### 4.2. 先前研究

> A mixed-initiative drawing tool that gives human agents direct control over the initial specification of drawings and indirect over their interactive evolution conducted by a computational agent

- Zhang et al. (2015) – DrawCompileEvolve
	- 使用者畫出初稿（直接控制）
	- AI 接手演化圖像（間接控制）
	- 人與電腦交替參與 → 為創作提供新變化

> The computational agent offers a sketch that should inspire them [...] conceptual shift (e.g., a plane as inspiration for a chair).

- Karimi et al. (2019)
	- 人畫草圖 → AI 給出靈感草圖（可以很跳 tone）
	- AI 使用「概念轉換（conceptual shift）」來啟發人類創造力
	- 例如：畫飛機 → AI 給出椅子的草圖 → 提高創意表現

> Design the computational agent to help the users in their tasks [...] receive completion suggestions using a text prompt.

- Ibarrola et al. (2022)
	- 使用者輸入提示文字（text prompt）
	- AI 給出完成建議（補全圖像）
	- 目標是讓 AI 輔助而非主導創作

> We created a sprite editor [...] allows a human agent to generate images procedurally

- 本研究
	- 使用者畫角色正面 → 拖進編輯器 → AI 幫你生出側面、背面
	- 可 快速交互、不受限制編輯、人機可以反覆互動
	- 每次互動都是一次「生成 → 編輯 → 再生成」的循環


## 5. 分工環節

paper：https://reurl.cc/VYgr7N

A 組員：負責背景與比較研究（文獻導向，較輕）
- 任務性質：背景鋪陳與脈絡整理，強調清楚脈絡與核心差異
- Section 1. Introduction
- Section 2. Related Work

B 組員：模型架構解釋
- 任務性質：技術細節＋網路架構理解，建議用圖輔助說明
- Section 3. Architecture Overview

C 組員：訓練與實驗方法
- 任務性質：說明如何驗證模型與實驗分析，建議以圖表、範例輔助。
- Section 4. Experiments
- Section 5. Results

D 組員：應用與未來展望
- 任務性質：偏應用面，適合擅長視覺或互動解說的組員。
- Section 6. Evaluation with All Sides
- Section 7. Sprite Editor 
- Section 8. Final Remarks


第H組：陳信嘉、朱晉賢、陳亭霓、陳維誠