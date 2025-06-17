Tuple 是一種不可變的有序集合

-   len()
-   indexing
-   count()
-   set()

```python
arr = ("助教", "丁丁","迪西", "拉拉", "小波")
print(arr)
print(type(arr)) # <class 'tuple'>
```

## 1. 遍歷所有元素

```python
arr = ('magic', 'xaab', 9099)      # 定義元組元素是字串與數字
for i in arr:
    print(i, end=" ")
```

## 2. list 與 tuple 互換

```python
arr_tuple = ("助教", "丁丁","迪西", "拉拉", "小波")
arr = list(arr_tuple)
arr.append("吉伊卡哇")
print(arr)
```

```python
arr_list = ["助教", "丁丁", "迪西", "拉拉", "小波"]
arr_tuple = tuple(arr_list)
print(arr_tuple)
```

## 3. 讀取 tuple & 切片

```python
arr = ("助教", "丁丁","迪西", "拉拉", "小波")
print(arr[0]) # 助教
```

```python
arr = ("助教", "丁丁","迪西", "拉拉", "小波")
print(arr[1:3]) # ('丁丁', '迪西')
print(arr[:2]) # ('助教', '丁丁')
print(arr[1:]) # ('丁丁', '迪西', '拉拉', '小波')
print(arr[-2:]) # ('拉拉', '小波')
print(arr[0:5:2]) # ('助教', '迪西', '小波')
```

## 4. tuple 解包與打包

![upgit_20240813_1723542222.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2024/08/upgit_20240813_1723542222.png)

### 4.1. Tuple Packing（元组打包）

```python
packed_tuple = 1, 'apple', 3.14
packed_tuple # packed_tuple = 1, 'apple', 3.14
```

### 4.2. tuple 解包與打包

```python
my_tuple = (1, 'apple', 3.14)
a, b, c = my_tuple # 數量不一樣會出錯
print("a:", a) # 1
print("c:", c) # 1
```
