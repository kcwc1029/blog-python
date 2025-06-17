使用JavaScript物件表示法編寫。
- json.loads()：JSON 格式的字串轉換為 Python 資料物件。
- json.dumps()：Python 資料物件轉換為 JSON 格式的字串，並返回字串。
- json.dump()：Python 資料物件轉換為 JSON 格式的字串，並直接寫入到指定的文件中。

```python
import json
# 物件轉json
dict = {
    "name":"TA", 
    "age":23,
    "city":"Tainan"
}
json_data = json.dumps(dict)
print(json_data , type(json_data))
# {"name": "TA", "age": 23, "city": "Tainan"} <class 'str'>
```

```python
import json
# json轉物件
json_data = '{"name": "TA", "age": 23, "city": "Tainan"}'
dict = json.loads(json_data)
print(dict) # {'name': 'TA', 'age': 23, 'city': 'Tainan'}
```

```python
import json

json_data = '{"name": "Alice", "age": 25, "city": "New York"}'

try:
    python_data = json.loads(json_data)
    print(python_data)
except json.JSONDecodeError as e:
    print("JSON 格式錯誤：", e) # {'name': 'Alice', 'age': 25, 'city': 'New York'}
```

