
import pandas as pd

# CSVファイルの読み込み
user_profile_data = pd.read_csv("user_profile_data.csv")

# カラム名のチェック
expected_columns = ["userId", "profile"]
if set(expected_columns) != set(user_profile_data.columns):
    raise ValueError("カラム名が一致しません。期待されるカラム: ['userId', 'profile']")

# userIdの範囲チェック
if not user_profile_data["userId"].between(1, 100).all():
    raise ValueError("無効なuserIdです。値は1から100までの整数である必要があります。")

# profileの文字数チェック
if user_profile_data["profile"].str.len().max() > 300:
    raise ValueError("プロファイルのテキストが300文字を超えています。")

print("フォーマットチェックが正常に完了しました。")
