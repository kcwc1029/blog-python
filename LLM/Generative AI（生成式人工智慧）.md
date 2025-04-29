定義： Generative AI 是一種 AI 系統，可以「生成新內容」，例如文字、圖片、音樂、影片、程式碼等。

```text
AI（人工智慧） 
  └─ ML（機器學習） 
      └─ DL（深度學習） 
          └─ GenAI（生成式 AI）
```

應用範圍：
- **Text Generation**：寫文章、劇本、詩
- **Image Creation**：繪圖、視覺設計
- **Code Writing**：生成與除錯程式碼
- **Music Composition**：創作音樂
- **Video、Audio 創作**
    
## 1. GenAI 的架構與運作流程（Architecture & Pipeline）

|步驟|說明|
|---|---|
|Prompt 輸入|使用者輸入要求，例如「幫我寫一首詩」|
|Foundation Model|使用大型預訓練模型（如 GPT、Stable Diffusion）|
|Generation Process|模型根據 prompt 進行生成|
|Output Creation|產出結果：文字、圖片、音訊、程式碼等|


## 2. Foundation Models（基礎模型）
- **Transformers**：處理語言序列（如 GPT）
- **Diffusion Models**：圖像生成（如 Stable Diffusion）
- **GANs（生成對抗網路）**：雙網對戰方式提升圖像品質
    

## 3. 支援機制：
- Attention 機制：選擇性注意重要資訊
- Progressive Generation：逐步生成細節
- Creative Competition：生成器與鑑別器對抗提升品質
    

## 4. GenAI 與傳統 AI 比較

| 項目   | 傳統 AI           | 生成式 AI (GenAI) |
| ---- | --------------- | -------------- |
| 核心功能 | 模式辨識、分類、預測      | 創造新內容          |
| 特徵處理 | 人工設計特徵、標註       | 模型自動理解與生成      |
| 任務導向 | 解決明確問題（判斷是狗還是貓） | 創作性輸出（生成一幅貓畫像） |
| 可解釋性 | 通常較高            | 多屬於黑盒（需更多解釋機制） |
|      |                 |                |


## 5. 挑戰與限制（Challenges）

|問題|說明|解法建議|
|---|---|---|
|幻覺（Hallucination）|生成錯誤資訊（不實、捏造）|加強訓練資料與人類審核|
|偏見（Bias）|模型可能重複訓練數據中的刻板印象|多元資料、審查機制|
|控制（Control）|難以完全掌控輸出風格或內容|加入引導詞、後處理技術|
|倫理（Ethics）|像是版權爭議、冒名生成等|法規建立、人機共管|

