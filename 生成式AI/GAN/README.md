# NOTE：

# 還沒整理完的東西

## 1. GANs (Generative Adversarial Networks)

GANs：全名 **Generative Adversarial Networks**，即生成對抗網路。
提出者：由 Ian Goodfellow 和蒙特婁大學（University of Montreal）的研究團隊於 **2014 年**提出。
屬性：屬於 **生成式 AI（Generative AI）** 的一種神經網路。
功能：能夠產生前所未有的新內容。
應用領域：圖像、音樂、語音、文字等多種領域。
架構：包含 Generator（生成器）與 Discriminator（判別器），兩者互相對抗的。
GANs 的目標：分佈的趨近（Distributions）

-   兩種機率分佈：
    -   **Pg**：Generator 輸出的資料分佈（隱式分佈，implicit distribution）。
    -   **Pr**：真實資料（如訓練集圖片）的分佈。
-   目標：
    -   讓 **Pg（生成的資料分佈）** 趨近於 **Pr（真實資料分佈）**。
    -   當兩者幾乎一致時，代表生成器能產生極為真實的資料。

## 2. GANs 的種類（Types of GANs）

常見的 GAN 變體：

-   **DCGANs**（Deep Convolutional GANs）：使用卷積神經網路（CNN）的 GAN。
-   **WGANs**（Wasserstein GANs）：改良訓練穩定性的 GAN。
-   **SRGANs**（Super Resolution GANs）：用於圖像超解析度。
-   **Pix2Pix**（Image-to-Image Translation）：圖像轉換任務。
-   **CycleGAN**（Cycle Generative Adversarial Network）：無需成對資料的圖像轉換。
-   **StackGAN**（Stacked GANs）：堆疊多層 GAN 以提升生成效果。
-   **ProGAN**（Progressive Growing GANs）：逐步成長式訓練的 GAN。
-   **StyleGAN**（Style-Based GANs）：強調風格控制（Style Transfer）的 GAN。
-   **VQGAN**（Vector Quantized GANs）：使用向量量化的方法提升生成表現。

其他有趣但較少為人知的延伸版：

-   **SGAN**
-   **InfoGAN**
-   **SAGAN**
-   **AC-GAN**
-   **GauGAN**
-   **GFP-GAN**

![upgit_20250427_1745741484.png|806x294](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250427_1745741484.png)

## 3. GAN 的流程步驟（Steps of a GAN）

1. **Generator** 接收隨機數值（噪聲）作為輸入，並生成一張假的圖片。
2. 這張生成圖片會與一批真實資料（例如：訓練集中的貓咪照片）一起送給 **Discriminator**。
3. **Discriminator** 根據資料特徵，計算每張圖片是真（真實）還是假（生成）的機率。
4. 輸出一個介於 0（假）到 1（真）之間的分數。

好的，這部分我來幫你整理成一份超清楚的「DCGAN 教學筆記」版本：  
（適合自學、教學或報告直接用）

---

## 4. DCGAN 教學筆記

-   **全名**：Deep Convolutional Generative Adversarial Network
-   **提出者**：Radford 等人在 **2015 年**發表。
-   **核心改進**： 以 **深度卷積神經網路（Deep Convolutional Networks, CNNs）** 取代原本 GAN 的全連接層（Fully Connected Layers）。
-   **優勢**： - CNN 擅長找出圖像中的局部特徵與空間相關性（spatial correlations）。 - 因此，**DCGAN 特別適合用於處理影像與影片資料**。 - 相較之下，傳統 GAN 更適合用於一般性的資料生成。
    原始論文：https://arxiv.org/abs/1511.06434

### 4.1. DCGAN 的基本運作原理

**Generator（生成器）**：

-   接收一組隨機噪聲向量（random noise），
-   經過多層反卷積（上取樣）處理，生成一張假圖片。
    **Discriminator（判別器）**：
-   接收真假圖片（包括 Generator 生成的和真實資料集中的圖片），
-   判斷每張圖片是真（1）還是假（0）。
    **使用損失函數（Loss Function）**：DCGAN 使用 **Binary Cross-Entropy（二元交叉熵）損失函數** 來優化兩個網路。
    **訓練特性**：Generator **無法直接接觸真實圖片**，只能透過 Discriminator 的反饋來學習。
    **最終目標**：
-   Generator 能夠騙過 Discriminator，產生出非常真實的假圖片。
-   Discriminator 能夠正確地辨識出圖片是真或假。

### 4.2. DCGAN 架構（Architecture）

![upgit_20250427_1745742584.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250427_1745742584.png)

```
[隨機噪聲 100維]
    ↓
[Project and Reshape]
    ↓
[Conv 1]（512 通道，Stride=2）
    ↓
[Conv 2]（256 通道，Stride=2）
    ↓
[Conv 3]（128 通道，Stride=2）
    ↓
[Conv 4]（64 通道，Stride=2）
    ↓
[生成圖片，單通道（如黑白圖片）]
```

每一層都會進行：

-   卷積（Conv）
-   批次正規化（BatchNorm）
-   激活函數（ReLU for Generator, LeakyReLU for Discriminator）

### 4.3. GAN 與 DCGAN 的成果比較

| 比較項目     | 傳統 GAN（200 次 Epoch） | DCGAN（100 次 Epoch）  |
| :----------- | :----------------------- | :--------------------- |
| 成果         | 圖片模糊且難以辨識       | 圖片清晰，數字特徵明顯 |
| 效率         | 訓練慢且效果差           | 訓練快且效果好         |
| 適合資料類型 | 泛用資料生成             | 影像/空間資料生成最佳  |

![upgit_20250427_1745742657.png|800x434](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250427_1745742657.png)

-   左：傳統 GAN 訓練 200 epoch 的手寫數字（MNIST）
-   右：DCGAN 訓練 100 epoch 的手寫數字（MNIST）

### 4.4. 先備知識：Dense 層（全連接層，Fully Connected Layer）

Dense 層是深度學習中一種非常基本且常見的神經網路層。

-   每一個「輸入的神經元」都會連接到每一個「輸出的神經元」。
    運作流程：
-   乘上權重矩陣（Weight Matrix）$z=W×x+b$
-   經過激活函數（Activation Function）$a=activation(z)$
    -   常見的 activation：`ReLU`、`Sigmoid`、`Softmax`、`Tanh` 等。
