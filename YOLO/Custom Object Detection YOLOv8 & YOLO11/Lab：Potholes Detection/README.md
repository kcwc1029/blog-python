### Lab：基於 YOLO8 路面坑挖檢測(Potholes Detection)

使用 YOLOv8 來進行路面坑洞（Potholes）的自動檢測，並比較了 自行標註數據 與 使用 Kaggle 現成標註數據 進行訓練的結果

## 數據預處理：資料標註與數據集來源

-   自行標註數據集：使用 Roboflow 手動標註坑洞，並匯出標註後的數據。
-   Kaggle 現成數據集：直接使用 Kaggle 上的標註數據，減少標註的時間成本。
-   使用 Data Augmentation（數據增強）來增加數據的多樣性
    -   旋轉（Rotation）
    -   亮度調整（Brightness Adjustment）
    -   鏡像翻轉（Flipping）

## 模型選擇與訓練

-   模型選擇：YOLOv8n（Nano 版）
-   訓練參數
    -   Batch Size：16
    -   Epochs：50
    -   Learning Rate：0.001

## 訓練過程

<span>
    <img src="https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989227.png" width="400px">
    <img src="https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989240.png" width="400px">
<span>

## 模型使用

<img src="./README DEMO/1740922498371.gif" width="400">

![upgit_20250303_1740989278.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989278.png)

## 模型評分結果

<span>
    <img src="https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989385.png" width="300">
    <img src="https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989391.png" width="300">
</span>

-   混淆矩陣（Confusion Matrix） 可以幫助我們了解錯誤分類的情況：

    -   TP（True Positive）：正確檢測到坑洞
    -   FP（False Positive）：誤判其他物體為坑洞
    -   FN（False Negative）：遺漏坑洞未檢測到

![upgit_20250303_1740989296.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/03/upgit_20250303_1740989296.png)

-   使用 mAP（Mean Average Precision）、Precision（精確率） 和 Recall（召回率） 來評估模型的表現：
    -   mAP（@0.5）：85.4%
    -   Precision（精確率）：83.2%
    -   Recall（召回率）：79.8%

## 程式碼

-   [Potholes_Detection + 自己用 roboflow 標註訓練](./Potholes_Detection_自己用roboflow標註訓練.ipynb)
-   [Potholes_Detection + kaggle 別人做好的資料集去做訓練](./Potholes_Detection_別人做好標籤.ipynb)
