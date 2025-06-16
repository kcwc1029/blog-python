## 1. 1.作業

請根據上課時所說的方式執行程式(不用執行本地端,本地端的部分可以跳過)，資料集部分請改用自己上網所找的資料集(CSV)餵進程式內,並且自行設計 prompt 進行問答。
請繳交程式碼以及執行畫面截圖(如 ppt 第 53 頁)

> NOTE：修改程式碼有使用 GPT 輔助。

## 2. CSV

```csv
// movie.csv
title,year,genre
Inception,2010,Science Fiction
Titanic,1997,Romance
```

## 3. 程式碼

```python
import ollama
import json

response = ollama.chat(
    model='llama3.2:latest',
    messages=[
        {'role': 'user', 'content': '1'}
    ]
)

# 建立可序列化的輸出字典
output = {
    "model": response.model,
    "created_at": response.created_at,
    "message": {
        "role": response.message["role"],
        "content": response.message["content"]
    },
    "done_reason": response.done_reason,
    "done": response.done,
    "total_duration": response.total_duration,
    "load_duration": response.load_duration
}

print(json.dumps(output, indent=4, ensure_ascii=False))
```

## 4. 輸出結果

![[image.png]]
