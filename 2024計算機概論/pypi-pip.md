> PyPI（Python Package Index） 和 pip（Python Package Installer） 之間的關係可以理解為「資料庫」和「下載工具」的關係。

## 1. PyPI（Python Package Index）

-   PyPI 是 Python 軟體包的官方託管和管理平台，相當於一個 Python 軟體包的「資料庫」或「倉庫」。
-   開發者可以將他們的 Python 套件上傳到 PyPI，這樣其他人就可以使用這些套件。
-   用戶可以在 PyPI 上搜尋、查看、下載各種 Python 套件，這些套件包括數據科學、網頁開發、機器學習等多種領域的工具。

## 2. pip（Python Package Installer）

-   pip 是用於從 PyPI 下載和安裝 Python 套件的工具，也可以管理已安裝的套件。
-   當我們使用命令 pip install package_name 時，pip 會自動連接到 PyPI，找到指定的套件並將其下載並安裝到本地環境中。
-   除了安裝，pip 還可以用於升級套件（pip install --upgrade package_name）和卸載套件（pip uninstall package_name）。

## 3. pip 常用指令

### 3.1. 檢查 pip 版本

```python
pip --version
```

### 3.2. 更新 pip 至最新版本

```
pip install --upgrade pip
```

### 3.3. 安裝套件：以 requests 套件為例

```
pip install requests
```

### 3.4. 安裝特定版本的套件

```
pip install requests==2.31.0
```

### 3.5. 更新套件至最新版本

```
pip install --upgrade requests
```

### 3.6. 列出已安裝的套件

```
pip list
```

### 3.7. 顯示特定套件的詳細資訊

```
pip show requests
```

### 3.8. 移除套件

```
pip uninstall requests
```
