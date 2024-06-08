import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main(data_folder, train_size, validation_size):
    # history.csvを読み込む
    df = pd.read_csv(data_folder + 'history.csv')

    # 全体のデータ数を取得
    total_count = len(df)

    # 学習データのサンプル数を指定の数にするためのtrain_sizeを計算
    train_size_ratio = train_size / total_count

    # データをランダムに分割
    train_data, temp_data = train_test_split(df, train_size=train_size_ratio, random_state=42)

    # 学習データに存在するユーザーのみを検証データに含める
    train_users = train_data['userId'].unique()
    val_data = temp_data[temp_data['userId'].isin(train_users)]

    # 検証データのサンプル数を指定の数にするためのvalidation_sizeを計算
    validation_size_ratio = validation_size / total_count

    # 検証データを絞り込む
    val_data = val_data.sample(n=validation_size, random_state=42)

    # 各データセットのサイズを取得
    train_count = len(train_data)
    val_count = len(val_data)

    # 結果を保存
    train_data.to_csv(data_folder + 'training.csv', index=False)
    val_data.to_csv(data_folder + 'validation.csv', index=False)

    # 各データセットのサイズを表示
    print(f"Total data count: {total_count}")
    print(f"Training data count: {train_count}")
    print(f"Validation data count: {val_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットを分割して保存します。")
    parser.add_argument("--data_folder", type=str, required=True, help="データを保存するフォルダのパス")
    parser.add_argument("--train_size", type=float, required=True, help="学習データのサイズ")
    parser.add_argument("--validation_size", type=int, required=True, help="検証データのサイズ")

    args = parser.parse_args()
    main(args.data_folder, args.train_size, args.validation_size)
