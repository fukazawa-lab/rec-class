import pandas as pd
import argparse
import string

def check_history_format(history_df):
    expected_columns = ["userId", "itemId", "rating"]
    if set(expected_columns) != set(history_df.columns):
        raise ValueError("history.csvのカラム名が一致しません。期待されるカラム: ['userId', 'itemId', 'rating']")

    user_counts = history_df['userId'].value_counts()
    if (user_counts < 4).any():
        raise ValueError("history.csvにおいて、4つ未満のデータしかないユーザが存在します。")

    if len(history_df) > 1000:
        raise ValueError("history.csvの行数が1000行を超えています。")

def check_metadata_format(metadata_df):
    expected_columns = ["itemId", "title", "genres", "release_date", "runtime", "overview"]
    if set(expected_columns) != set(metadata_df.columns):
        raise ValueError("metadata.csvのカラム名が一致しません。期待されるカラム: ['itemId', 'title', 'genres', 'release_date', 'runtime', 'overview']")

def check_user_profile_data_format(user_profile_data_df, history_df):
    expected_columns = ["userId", "profile"]
    if set(expected_columns) != set(user_profile_data_df.columns):
        raise ValueError("user_profile_data.csvのカラム名が一致しません。期待されるカラム: ['userId', 'profile']")

    if not set(history_df['userId']).issubset(set(user_profile_data_df['userId'])):
        raise ValueError("history.csvのすべてのuserIdがuser_profile_data.csvに含まれていません。")

    if user_profile_data_df["profile"].str.len().max() > 300:
        raise ValueError("user_profile_data.csvのプロファイルのテキストが300文字を超えています。")

    invalid_profile = user_profile_data_df["profile"].apply(lambda x: any(char not in string.printable for char in x)).any()
    if invalid_profile:
        raise ValueError("user_profile_data.csvのプロファイルに英語以外の文字が含まれています。")

def check_item_ids_in_metadata(history_df, metadata_df):
    if not set(history_df["itemId"]).issubset(set(metadata_df["itemId"])):
        raise ValueError("history.csvのitemIdがmetadata.csvにすべて含まれていません。")

def main(history_file, metadata_file, user_profile_data_file):
    # CSVファイルの読み込み
    history_df = pd.read_csv(history_file)
    metadata_df = pd.read_csv(metadata_file)
    user_profile_data_df = pd.read_csv(user_profile_data_file)

    # フォーマットチェック
    check_history_format(history_df)
    check_metadata_format(metadata_df)
    check_user_profile_data_format(user_profile_data_df, history_df)
    check_item_ids_in_metadata(history_df, metadata_df)

    print("すべてのフォーマットチェックが正常に完了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3つのCSVファイルのフォーマットチェックを行います。")
    parser.add_argument("history_file", type=str, help="history.csvファイルのパス")
    parser.add_argument("metadata_file", type=str, help="metadata.csvファイルのパス")
    parser.add_argument("user_profile_data_file", type=str, help="user_profile_data.csvファイルのパス")
    
    args = parser.parse_args()
    main(args.history_file, args.metadata_file, args.user_profile_data_file)
