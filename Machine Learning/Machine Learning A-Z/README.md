## 簡單線性回歸

![upgit_20241219_1734602115.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241219_1734602115.png)

-   [simple_linear_regression.ipynb](./線性回歸/simple_linear_regression.ipynb)

### 1. 普通最小平方法 (Ordinary Least Squares, OLS)

![upgit_20241219_1734602354.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241219_1734602354.png)

## 2. 多元線性回歸(Multiple_Linear_Regression)

![upgit_20241219_1734608021.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241219_1734608021.png)

![upgit_20241219_1734608030.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241219_1734608030.png)

![upgit_20241219_1734608511.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241219_1734608511.png)

-   [Multiple_Linear_Regression.ipynb](./多元線性回歸/Multiple_Linear_Regression.ipynb)

## 多項式迴歸(Polynomial Linear Regression)

-   屬於非線性的模型，用於預測具有非線性關係的因變量(y)。它通過加入自變量的高次項來建模複雜的關係。
-   [polynomial_linear_regression.ipynb](./多項式迴歸/polynomial_linear_regression.ipynb)

## SVR(Support Vector Regression)變體支援向量機

![upgit_20241220_1734692979.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734692979.png)

-   SVR (Support Vector Regression) 是支援向量機 (SVM, Support Vector Machine) 的一種變體，用於處理迴歸問題。
-   非線性核 SVR (Non-linear Kernel SVR)：基於核函數的擬合方式，特別適合非線性迴歸問題
-   [support_vector_regression.ipynb](./SVR/support_vector_regression.ipynb)

## 決策樹(Decision Tree)

![upgit_20241220_1734696140.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734696140.png)

![upgit_20241220_1734696191.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734696191.png)

-   [決策樹回歸：decision_tree.ipynb](./決策樹回歸/decision_tree.ipynb)

![upgit_20241223_1734961612.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734961612.png)

![upgit_20241223_1734961628.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734961628.png)

-   [決策樹分類：decision_tree.ipynb](./決策樹分類/decision_tree.ipynb)

## 隨機森林(Random Forest)

-   隨機森林是一種基於決策樹的集成學習方法。
-   它通過構建多個決策樹並將它們的結果進行平均 (迴歸) 或投票 (分類)，來提升模型的準確性和穩健性。
-   STEP 1: 從訓練數據集中隨機選取 K 個數據點
-   STEP 2: 使用這些 K 個數據點構建一棵決策樹
-   STEP 3: 選擇要構建的樹的數量平均，重複 STEP1 STEP2
-   [random_forest.ipynb](./隨機森林回歸/random_forest.ipynb)

## R 平方

![upgit_20241220_1734700154.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734700154.png)

## 邏輯回歸(Logistic Regression)

-   用於分類問題，預測類別
-   使用 Sigmoid 函數 將線性方程的輸出轉換為概率
-   預測結果為概率（介於 0 和 1 之間）

![upgit_20241220_1734700524.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734700524.png)

![upgit_20241220_1734700621.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734700621.png)

![upgit_20241220_1734700469.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734700469.png)

-   [logistic_regression.ipynb](./邏輯回歸/logistic_regression.ipynb)

## KNN (K-Nearest Neighbors)

-   是一種分類(Classification)算法。
-   它不需要訓練過程。
-   給定一個數據點，找到距離它最近的 K 個鄰居，然後根據這些鄰居的標籤進行預測。
-   工作流程
    -   計算待測數據點與所有訓練數據的距離。
    -   找到距離最近的 K 個鄰居。
    -   根據鄰居的標籤進行分類或回歸：
        -   分類：選擇 K 個鄰居中出現最多的標籤（多數決）。
        -   回歸：對 K 個鄰居的標籤取平均值。

![upgit_20241220_1734703075.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734703075.png)

![upgit_20241220_1734703090.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734703090.png)

![upgit_20241220_1734703099.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241220_1734703099.png)

-   [knn.ipynb](./KNN/knn.ipynb)

## 支持向量機 (Support Vector Machine, SVM)

-   SVM 是一種監督式機器學習算法，
-   用於解決分類和回歸問題，特別適合小樣本、高維數據的應用場景。

![upgit_20241223_1734944604.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734944604.png)

-   超平面 (Hyperplane)
    -   SVM 的目標是找到一個最佳超平面，將不同類別的數據分開。
    -   對於二分類問題，超平面是一條分隔數據的線（2D 空間）或一個面（3D 空間及更高）。
-   最大間隔 (Maximum Margin)
    -   SVM 通過最大化類別之間的間隔來選擇超平面。
    -   間隔（Margin）是從超平面到最接近數據點（支持向量）的距離。
    -   最大間隔能提高模型對新數據的泛化能力。
-   支持向量 (Support Vectors)
    -   支持向量是距離超平面最近的數據點。
-   [support_vector_machine](./支持向量機/support_vector_machine.ipynb)

## kernal SVM

-   Kernel SVM 是 SVM 的擴展，用於解決非線性分類問題。
-   透過 核函數 (Kernel Function)，數據從低維空間隱式映射到高維空間，使其在高維空間中線性可分。
-   [kernal_SVM.ipynb](./Kernel SVM/kernal_SVM.ipynb)

## 貝氏分類器(Naive Bayes Classifier)

![upgit_20241223_1734952660.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734952660.png)

-   貝氏分類器 是一種基於貝氏定理的機器學習算法，
-   特別適合用於文本分類（如垃圾郵件過濾）、情感分析等任務。它簡單、高效，適合處理高維數據。

![upgit_20241223_1734952889.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734952889.png)

-   [naive_bayes_classifier.ipynb](./貝氏分類器/naive_bayes_classifier.ipynb)

## cluster(聚集)

![upgit_20241223_1734963017.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734963017.png)

![upgit_20241223_1734963027.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241223_1734963027.png)

## K-means

-   是一種無監督學習演算法，用於資料的聚類分析。它會將資料分成
    𝑘 個群集 (clusters)，每個群集都有自己的中心點 (centroid)。

![upgit_20241224_1735044022.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241224_1735044022.png)

![upgit_20241224_1735045118.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241224_1735045118.png)

-   [k_means.ipynb](./k-means/k_means.ipynb)

## 層次聚類(Hierarchical Clustering)

-   Hierarchical Clustering (層次聚類) 是一種無監督學習演算法。
-   來將數據分成多個群集。
-   層次聚類的結果通常以樹狀圖（Dendrogram） 表示，顯示數據的合併過程和層次結構。

![upgit_20241224_1735046516.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241224_1735046516.png)

![upgit_20241224_1735046379.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241224_1735046379.png)

-   [hierarchical_clustering.ipynb](./層次聚類/hierarchical_clustering.ipynb)

## 關聯規則學習(Associated Rules Learning)

-   啤酒跟尿布的故事
-   關聯式學習 (Association Rule Learning) 是一種無監督學習技術。
-   用於發現數據集中項目之間的關聯規則。
-   經常用於分析購物籃數據，找出哪些商品經常一起購買。

-   Apriori
    -   Apriori 則是 "挖掘項目之間的條件規則"
    -   支持度、置信度、提升度 (Support, Confidence, Lift)
    -   適合稀疏數據集（交易項目較少）
    -   通用，計算規則需更多步驟
    -   [apriori.ipynb](./ARL：apriori/apriori.ipynb)

![upgit_20241224_1735049646.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241224_1735049646.png)

-   Eclat
    -   Eclat 專注於 "發現常見組合"
    -   只看支持度 (Support)
    -   內存依賴高，對稠密數據效率高（交易項目較多）
    -   程式碼 nono

## Reinforcement Learning

-   有兩種：
    -   Upper Confidence Bound (UCB)
    -   Thompson Sampling

### Multi-Armed Bandit Problem（多臂老虎機問題）

-   假設你面前有多台老虎機（每台老虎機都有不同的回報率，這些回報率是未知的），你需要決定如何拉動不同的老虎機，來最大化你的總收益。
-   核心挑戰
    -   探索（Exploration）：試探新的老虎機以獲取更多資訊，確定它是否能帶來更高的回報。
    -   利用（Exploitation）：利用目前已知回報率最高的老虎機來獲取最大的收益。

![upgit_20241225_1735129312.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735129312.png)

![upgit_20241225_1735129318.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735129318.png)

![upgit_20241225_1735133057.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735133057.png)

## Upper Confidence Bound (UCB)

-   確定性 (Deterministic)：
    -   每個回合選擇廣告時，計算一個確定的「上置信界限」值，並選擇該值最大的廣告。
    -   公式中會明確考慮「平均回報率」和「不確定性」。
-   每回合需要更新：
    -   每一輪結束後，必須立刻更新每個廣告的數據 (如點擊次數、總展示次數)。
    -   適合需要即時更新模型的場景。
-   直觀理解：想像每個廣告的表現如同罐子裡的水位，UCB 選擇「目前表現最好或可能更好」的廣告，確保上置信界限最大化。

![upgit_20241225_1735130588.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735130588.png)

-   [ucb.ipynb](./UCB/ucb.ipynb)

## Thompson Sampling

-   機率性 (Probabilistic)：
    -   使用 Beta 分布來表示每個廣告的點擊率的「不確定性」。
    -   每次從每個廣告的分布中抽取一個值，選擇值最大的廣告。
-   可以處理延遲回饋：即使某些回饋數據需要時間才能到達，Thompson Sampling 仍能有效處理，因為它基於分布而非確定值。
-   更好的實驗證據 (Empirical Evidence)：在許多實驗中，Thompson Sampling 表現優於 UCB，尤其是在真實環境中具有不確定性時。
-   直觀理解：想像每個廣告的表現是一個「鐘型分布」，分布越靠右代表可能的點擊率越高。Thompson Sampling 每次從分布中抽取值，動態平衡「探索」和「利用」。

![upgit_20241225_1735132985.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735132985.png)

-   [thompson_sampling.ipynb](./湯普森取樣/thompson_sampling.ipynb)

## NLP

-   將機器學習模型應用於文字和語言。自然語言處理的重點在於教導機器理解口語和書面所說的內容。
-   口述內容轉換為文字時，就是 NLP 演算法在運作的時候。
-   也可以在文字評論上使用 NLP 來預測評論是好是壞。
-   在文章上使用 NLP 來預測您試圖分割的文章的某些類別。
-   大部份的 NLP 演算法都是分類模型，
    -   邏輯回歸 (Logistic Regression)、
    -   奈夫貝伊 (Naive Bayes)、
    -   以判定樹為基礎的 CART 模型
    -   與判定樹相關的最大熵 (Maximum Entropy)
    -   隱馬可夫模型 (Hidden Markov Models)。

### NLP 的類型

![upgit_20241225_1735133599.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/12/upgit_20241225_1735133599.png)

-   傳統 NLP 與深度學習 NLP 的區別
-   傳統 NLP (Traditional NLP):
    -   使用詞袋模型 (BoW)、TF-IDF 等表示文本。
    -   基於統計或機器學習模型，例如 Naive Bayes、SVM。
    -   特徵工程是核心，需要手動設計特徵。
-   深度學習 NLP (DLNLP):
    -   使用深度學習模型 (如 RNN、Transformer) 自動學習特徵。
    -   更適合處理大規模數據。
    -   支持預訓練模型 (如 BERT、GPT) 的微調。

### Bag-of-Words (BoW)

-   Bag-of-Words (BoW) 是一種在自然語言處理 (NLP) 和文本分析中非常常見的技術
-   用於將文本數據轉換為數字特徵，方便機器學習模型處理。

-   [nlp.ipynb](./NLP/nlp.ipynb)

## 人工神經網路(artificial neural network)

### 神經元(neural)

![upgit_20250101_1735733318.png](https://r aw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250101_1735733318.png)

### Gradient Descent

![upgit_20250101_1735736683.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250101_1735736683.png)

### 批量梯度下降 (Batch Gradient Descent) 和 隨機梯度下降 (Stochastic Gradient Descent, SGD)

#### 隨機梯度下降 (Stochastic Gradient Descent, SGD)

-   逐條數據樣本計算梯度並立即更新權重
-   每次計算僅需要一條數據，速度快，對大數據集更高效。
-   權重更新頻繁，能夠快速適應數據的變化。
-   因為每次只使用一條數據，梯度更新方向可能不穩定，容易在最優解附近震盪

![upgit_20250101_1735736158.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250101_1735736158.png)

#### 批量梯度下降 (Batch Gradient Descent)

-   計算整個數據集的平均梯度後再更新權重。
-   針對數據的全局信息進行學習，方向更加穩定。
-   適合小數據集，因為可以準確地找到損失函數的最優方向。
-   如果數據集很大，每次計算都會很慢，因為需要處理整個數據集。
-   內存消耗大。

-   [人工神經網路](./人工神經網路/ann.ipynb)

## 卷積神經網路(CNN)

![upgit_20250102_1735821470.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735821470.png)

### 多重視角的特徵提取

-   幻象(一個老太太/一個往後看的女生)

![upgit_20250102_1735820196.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735820196.png)

-   幻象(鴨子/兔子)
    ![upgit_20250102_1735820345.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735820345.png)

### 感知錯誤與特徵干擾

![upgit_20250102_1735820462.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735820462.png)

### convolution

![upgit_20250102_1735822371.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735822371.png)
![upgit_20250102_1735822946.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735822946.png)

-   小工具：https://convolution-solver.ybouane.com/
    -   Kernel Size
    -   Padding (填充)
    -   Dilation (膨脹)：擴展卷積核內部的間距
    -   Stride (步幅)
    -   Transposed Convolution：是否使用轉置卷積 (反卷積) 進行操作

### ReLU layer

-   卷積層操作的結果，生成了多個 特徵圖 (Feature Maps)
-   每個特徵圖可能包含負值和正值，而 ReLU 的主要功能是將所有的負值轉換為 0，保持正值不變。
    ![upgit_20250102_1735823225.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735823225.png)

### Max pooling

![upgit_20250102_1735823529.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735823529.png)

-   用於縮小特徵圖的尺寸，同時保留重要的特徵資訊。
-   縮小特徵圖尺寸：減少計算量和存儲需求，讓模型更高效。
-   保留關鍵特徵：只取區域內的最大值，保留最重要的激活值。
-   減少過擬合：去除不重要的細節，增加模型的泛化能力。

### flattening

![upgit_20250102_1735823829.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735823829.png)

![upgit_20250102_1735825071.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735825071.png)

![upgit_20250102_1735825176.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735825176.png)

### Softmax & Cross-Entropy Loss

![upgit_20250102_1735828058.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250102_1735828058.png)

### CNN 視覺化小工具

-   https://poloclub.github.io/cnn-explainer/
-   https://adamharley.com/nn_vis/cnn/3d.html

## 主成分分析(PCA)

-   主成分分析 (Principal Component Analysis, PCA) 是一種常用的降維技術。
-   用於在保留資料主要資訊的前提下，減少特徵的數量。
-   相關應用
    -   降維： 減少數據特徵數量，降低計算成本。
    -   可視化： 將高維數據投影到 2D 或 3D 空間，方便觀察和分析。
    -   去除噪聲： 去除不重要的成分，提升模型性能。
    -   特徵選擇： 聚焦於最重要的變化方向。

### PCA 的步驟

-   標準化數據：PCA 對數據的縮放非常敏感，因此需要將數據標準化，使其平均值為 0，標準差為 1。
-   計算共變異數矩陣：共變異數矩陣用來表示特徵之間的關係，幫助找出數據的主要變化方向。
-   計算特徵值和特徵向量
-   選擇前幾個具有最大特徵值的主成分，這些組合保留了數據中最重要的信息
-   將原始數據映射到選定的主成分上，從而完成降維。

-   [PCA](./PCA/pca.ipynb)

## 線性判別分析 (Linear Discriminant Analysis, LDA)

-   LDA 是一種 降維技術，與主成分分析 (PCA) 相似，但目標不同
-   PCA ：
    -   找到數據中變異量最大的方向，與標籤無關
    -   尋找變異量最大的軸
-   LDA 的目標
    -   將數據從高維度空間投影到低維空間。
    -   最大化類別之間的分離，同時最小化類別內的變異。
    -   尋找能最大化類別間分離的軸。
-   PCA 是非監督式降維技術，而 LCA 是非監督式降維技術

### LCA 的步驟

-   計算每個類別的平均向量：對每個類別，計算其在每個特徵上的平均值（類似於中心點）。
-   計算類內散佈矩陣和類間散佈矩陣：
    -   類內散佈矩陣：度量每個類別內數據點的分佈情況。
    -   類間散佈矩陣：度量各類別的平均向量之間的分佈情況。
-   求解特徵值和特徵向量：通過最大化 類間散佈 / 類內散佈 的比率，計算出線性判別向量。
-   選擇主軸進行投影：根據特徵值的大小，選擇 k 個最大的特徵向量，作為新的投影軸。
-   投影數據到低維空間：使用選擇的特徵向量將原始數據映射到低維空間。

-   [LDA](./LDA/lda.ipynb)

## Kernel PCA

-   PCA (Principal Component Analysis)是將將高維數據映射到低維空間，同時保留數據的最大變異量
-   僅適用於線性數據。如果數據的結構是非線性的（例如，數據分佈呈曲面或非線性模式），PCA 無法很好地表示。
-   Kernel PCA 通過內核方法 (Kernel Method)，將數據映射到高維空間，然後在高維空間進行 PCA。
-   方法：
    -   使用核函數將數據從原始空間轉換到高維空間，捕捉非線性特徵。
    -   在高維空間中應用 PCA，找到主成分。
-   能夠處理非線性數據分佈。例如，如果數據呈現曲面、圓形或其他非線性結構，Kernel PCA 可以更好地捕捉數據特徵。

-   [Kernel PCA](./Kernel%20PCA/kernel_pca.ipynb)

## Bias-Variance Tradeoff (偏差-方差權衡)

-   Bias-Variance Tradeoff 是機器學習中用於理解模型表現的重要概念。
-   涉及到如何平衡模型的 偏差 (Bias) 和 方差 (Variance)，以實現最佳的預測性能。
-   Bias (偏差)：
    -   偏差是模型的預測值與真實值之間的差距
    -   通常來自模型過於簡單（欠擬合）的情況。
    -   偏差高的模型難以捕捉數據的模式，導致預測不準確(像是線性回歸對於非線性數據)
-   Variance (方差)：
    -   方差是模型對不同訓練數據集的敏感度
    -   方差高的模型對於訓練數據的微小變化過於敏感，導致泛化能力差。
    -   例子：決策樹深度過大，過度擬合訓練數據。

![upgit_20250109_1736429442.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250109_1736429442.png)

-   High Bias, Low Variance (高偏差，低方差)
    -   模型過於簡單，無法捕捉數據的複雜模式（欠擬合）
    -   預測點分佈偏離中心（無法準確描述目標）
    -   例如線性模型用於非線性數據集
-   High Bias, High Variance (高偏差，高方差)
    -   預測結果既分散又偏離目標中心。
    -   極端不適合的模型或設置（非常差的模型）。
-   Low Bias, High Variance (低偏差，高方差)
    -   模型過於複雜，過度擬合訓練數據（過擬合）
    -   模型對訓練數據表現良好，但泛化能力差(例如深度過大的決策樹。)
-   Low Bias, Low Variance (低偏差，低方差)
    -   理想情況，模型既能準確捕捉數據模式（低偏差），又能很好地泛化到測試數據（低方差）。

## Model Selection & Boosting

-   是一種在數據有限的情況下是用的評估模型性能的技術
-   K-Fold Cross-Validation 是將數據集分為 K 等份（folds），並進行多次訓練和測試的一種驗證方法。
-   更穩健的性能評估： 通過多次訓練和測試，減少了數據分割帶來的偏差。
-   充分利用數據： 每個數據點都會在訓練集和測試集中出現一次，不浪費數據。
-   避免過度擬合： 對不同測試集進行評估，可以更好地檢查模型的泛化能力。

### 使用具體過程

-   數據分割
    -   將數據集劃分為 K 等份。
    -   每次選擇其中一份作為測試集，剩下的 K−1 份作為訓練集
-   多次訓練：模型將被訓練 K 次，每次都會用不同的數據部分作為測試集
-   結果平均：每次驗證的結果（如準確率、F1 分數）會被記錄下來，最後取平均作為模型的最終評估結果。

![upgit_20250109_1736429250.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250109_1736429250.png)

![upgit_20250109_1736429268.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/01/upgit_20250109_1736429268.png)

-   [使用 K-Fold Cross-Validation 在 SVM 上進行驗證](./k-Fold%20Cross/SVM_with_k-fold.ipynb)

## XGBoost

-   XGBoost (eXtreme Gradient Boosting) 是一種基於梯度提升 (Gradient Boosting) 的強大且高效的機器學習算法
-   主要用於回歸和分類問題。它以其快速的計算速度和優秀的預測性能受到廣泛使用。
-   基於決策樹的提升方法，通過結合多個弱學習器（通常是決策樹）來形成一個強大的預測模型。
-   核心特性：
    -   提升方法 (Boosting)：逐步構建模型，每一步改進前一步的預測結果。通過學習上一輪預測的殘差來最小化損失。
    -   通過正則化項控制模型的複雜性，防止過擬合
    -   支持多線程運行，實現快速計算
    -   XGBoost 內置處理缺失值的機制，不需要預先填補
-   安裝：pip install xgboost
-   XGBoost 的主要參數

    -   n_estimators: 樹的數量。
    -   learning_rate: 學習率，決定每棵樹對最終預測的影響。
    -   max_depth: 每棵樹的最大深度，控制模型的複雜性。
    -   subsample: 隨機取樣比例，用於防止過擬合。
    -   colsample_bytree: 每棵樹使用的特徵數比例。
    -   objective: 損失函數，如 binary:logistic (二分類) 或 reg:squarederror (回歸)。

-   [XGBoost](./XGBoost/XGBoost.ipynb)
