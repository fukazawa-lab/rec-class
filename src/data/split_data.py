import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main(data_folder, train_size):
    # history.csvを読み込む
    df = pd.read_csv(data_folder + 'history.csv')

    # 学習データのサンプル数を1000にするためのtrain_sizeを計算
    train_size = train_size / len(df)

    # データをランダムに分割
    train_data, temp_data = train_test_split(df, train_size=train_size, random_state=42)

    # 学習データに存在するユーザーのみを検証データに含める
    train_users = train_data['userId'].unique()
    val_data = temp_data[temp_data['userId'].isin(train_users)]

    # 結果を保存
    train_data.to_csv(data_folder + 'training.csv', index=False)
    val_data.to_csv(data_folder + 'validation.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットを分割して保存します。")
    parser.add_argument("--data_folder", type=str, required=True, help="データを保存するフォルダのパス")
    parser.add_argument("--train_size", type=float, required=True, help="学習データのサイズ (0.0より大きく1.0以下の値)")

    args = parser.parse_args()
    main(args.data_folder, args.train_size)
