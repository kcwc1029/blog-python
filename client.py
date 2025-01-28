import pandas as pd

# 擴展數據範例
data = {
    "Unit": ["S1", "S2", "S3", "S4", "S1", "S2", "S3", "S4", "S1", "S2", "S3", "S4"],
    "Conversation": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2", "C3", "C3", "C3", "C3"],
    "Content": [
        "我覺得我們應該先討論這個問題。", "好主意！我們應該怎麼開始？",
        "我認為這個解法可能有問題。", "我們可以嘗試另一種方法。",
        "有沒有人有其他建議？", "或許可以參考之前的案例。",
        "這樣可能會更有效。", "我們需要更詳細的數據支持。",
        "接下來應該如何繼續？", "我們可以分工合作完成這部分。",
        "這部分我可以負責撰寫報告。", "需要確認一下資料是否完整。"
    ],
    "Code": ["提問", "回答", "澄清", "提問", "提問", "回答", "澄清", "提問", "提問", "回答", "回答", "澄清"]
}

df = pd.DataFrame(data)

# 保存數據
file_path = "./ENA_expanded_data.csv"
df.to_csv(file_path, index=False)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Expanded ENA Dataset", dataframe=df)

# file_path
