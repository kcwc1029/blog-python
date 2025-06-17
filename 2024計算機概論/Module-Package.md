以下是python常見的module，外部module會使用pip去安裝

![upgit_20240817_1723873329.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/08/upgit_20240817_1723873329.png)

## 1. 自己建立模組
- 如果要導入多組函數 from make_food import make_icecream, make_drink
- 如果要導入所有函數 from make_food import *

```python
def make_icecream(*args):
    """這邊寫關於函數的說明
    Args:
        toppings (string): 各種配料
    Return:
        none
    """

    print("這個冰淇淋所加配料如下")
    for topping in args:
        print("--- ", topping)


def make_drink(size, drink):
    """這邊寫關於函數的說明
    Args:
        size (string): 
        drink (string): 各種飲料
    Return:
        none
    """
    print("所點飲料如下")
    print("--- ", size)
    print("--- ", drink)



make_icecream('草莓醬')
make_icecream('草莓醬', '葡萄乾', '巧克力碎片')
make_drink('large', 'coke')
```
## 2. 應用自己寫的module
我覺得我知還會再用到這個func，所以我把它放到module

```python
%%writefile make_food.py

def make_icecream(*toppings):
    """這邊寫關於函數的說明
    Args:
        toppings (string): 各種配料
    Return:
        none
    """

    print("這個冰淇淋所加配料如下")
    for topping in toppings:
        print("--- ", topping)


def make_drink(size, drink):
    """這邊寫關於函數的說明
    Args:
        size (string): 
        drink (string): 各種飲料
    Return:
        none
    """
    print("所點飲料如下")
    print("--- ", size.title())
    print("--- ", drink.title())
```

> 要使用的地方把它引入 from module_name import func_name

```python
from make_food import make_icecream

make_icecream('草莓醬', '葡萄乾', '巧克力碎片')

# 這個冰淇淋所加配料如下
# ---  草莓醬
# ---  葡萄乾
# ---  巧克力碎片
```

## 3. module 使用別稱as
可以分為：
- 幫module取名
- 幫module的func取名
```python
from make_food import make_icecream as m

m('草莓醬', '葡萄乾', '巧克力碎片')
```

```python
import make_food  as m

m.make_icecream('草莓醬', '葡萄乾', '巧克力碎片')
```

## 4. random module

![upgit_20241021_1729492699.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/10/upgit_20241021_1729492699.png)

```python
import random

random.seed(24)

print(random.randint(1, 10)) # 隨機整數
print(random.random()) # 0-1 隨機浮點數
print(random.uniform(1,3)) # 範圍間隨機浮點數
print(random.choice(["a", "b", "c"])) # 範圍間隨機選
print(random.sample(["a", "b", "c"], 2)) # 範圍間隨機選要求數量

# 隨機排列
arr = ["a", "b", "c"]
random.shuffle(arr)
print(arr)
```

## 5. time module

![upgit_20241021_1729492730.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/10/upgit_20241021_1729492730.png)

```python
import time

print(time.time()) # 1970.01.01到現在秒數
print(time.ctime()) # 目前系統時間

print(time.localtime()) # 返回tuple結構時間
print(time.localtime().tm_year) # 2025
print(time.localtime()[0]) # 2052
```

## 6. sys module
可以了解python shell的訊息。

```python
import sys

# 顯示python訊息
print(sys.version)
print(sys.version_info) # 比較結構化
```

```python
# 列出module所在的路徑
import sys
for i in sys.path:
    print(i)
```

## 7. stdin(standard input 的縮寫)
```python
import sys

# 要額外開+terminal
print("請輸入字串")
msg = sys.stdin.readline()
print(msg)

# 規定要取幾個字元
print("請輸入字串")
msg = sys.stdin.readline(3)
print(msg)
```

## 8. stdout(standard output 的縮寫)

```python
import sys

# 要額外開+terminal
sys.stdout.write("122323")
```

## 9. 將自己寫的class建立在module內
類別也可以用module的方式去處理
```python
%%writefile make_band.py
class Banks():
    def __init__(self, name):    
        self.__name = name
        self.__balance = 0
        self.__title = "Bank"

    def save_money(self, money): # 設計存款方法
        self.__balance += money
        print("存款 ", money, " 完成")

    def withdraw_money(self, money): # 設計提款方法
        self.__balance -= money             
        print("提款 ", money, " 完成")

    def get_balance(self):   # 獲得存款餘額
        print(self.__name.title(), " 目前餘額: ", self.__balance)

    def bank_title(self):  # 獲得銀行名稱
        return self.__title


class Tainan_bank(Banks):
    def __init__(self, name):
        self.__title = "Tainan bank"
    def bank_title(self):  # 獲得銀行名稱
        return self.__title
```

```python
from make_band import Banks, Tainan_bank

TA = Banks('James')  
print("TA's banks = ", TA.bank_title())  # 列印銀行名稱
TA.save_money(500)                   # 存錢
TA.get_balance()                     # 列出存款金額
hung = Tainan_bank('Hung')             # 定義Shilin_Banks類別物件
print("hung's banks  = ", hung.bank_title())   # 列印銀行名稱
```

## 10. module searching
針對模組放的路徑，可以有一下選擇：
- 放在該專案資料夾裡面的 sys.path
- 跟 main program 放在一起
- 放在任何地方，再去更改 sys.path
```python
import sys

print(*sys.path, sep="\n") # 電腦會根據以下每一個路徑去找module
```

更改sys.path
```python
sys.path.append("要增加的路徑")
```


## 11. if name == "main"
- 在 Python 中，每個檔案都可以被視為一個模組，當模組被運行時，Python 會賦予一個特殊變數 __name__。
- 如果該檔案是被直接運行的（比如用 python filename.py 來執行），那麼 __name__ 的值會被設為 "__main__"。
- 如果這個檔案是被作為模組導入的，那麼 __name__ 的值會是這個模組的名字（也就是檔案名）。

```python
%%writefile calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

if __name__ == "__main__":
    # 這些程式碼只在這個檔案直接運行時執行
    print("Testing add function:", add(2, 3))           # 5
    print("Testing subtract function:", subtract(5, 2)) # 3
```

## 12. namespace
其實就是前面提到 LEGB 規則的延伸，程式在找變數（函數），會先從區域命名空間（local namespace）去找，接著是全域命名空間（global namespace），最後是 built-in（內建函式庫）

```python
# 輸出內建函式庫
print(__builtins__) # <module 'builtins' (built-in)>

# 輸出所有函數名稱
print(*dir(__builtins__), sep="\n")
```

## 13. package
package 是一種包含很多 python 模組的字典
- PyPI：全名 Python Package Index，是指 Python 生態系統中的一個中央軟件存儲庫。
- pip：安裝第三方庫的指令：

```shell
# TODO: pip相關指令
pip install <package_name>
pip install -upgrade <package_name>

# 安裝指定版本
pip install <package_name> == <指定版本>

# 確認目前安裝所有模組及版本
pip freeze
```
