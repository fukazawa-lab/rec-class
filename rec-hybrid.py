import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

def calculate_rmse(file_svd, file_cbf):
    # CSVファイルの読み込み
    validation_predictions_svd = pd.read_csv(file_svd)
    validation_predictions_cbf = pd.read_csv(file_cbf)

    validation_predictions_cbf.rename(columns={'rating': 'rating_cbf'}, inplace=True)
    validation_predictions_svd.rename(columns={'rating': 'rating_svd'}, inplace()

    # userId_movieIdでデータを突き合わせる
    merged_df = pd.merge(validation_predictions_svd, validation_predictions_cbf, on='userId_movieId')

    # αを0から1まで0.1刻みで変化させる
    alphas = np.arange(0, 1.1, 0.1)

    # RMSEを格納するリスト
    rmses = []

    # true_ratingを抽出
    true_ratings = merged_df['true_rating']

    # 各αに対して重みづけ和を計算し、RMSEを算出
    for alpha in alphas:
        # 重みづけ和の計算
        weighted_predictions = alpha * merged_df['rating_cbf'] + (1 - alpha) * merged_df['rating_svd']

        # RMSEの計算
        rmse = np.sqrt(mean_squared_error(true_ratings, weighted_predictions))
        rmses.append(rmse)
        print(f"Alpha: {alpha:.1f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RMSE for different alpha values.")
    parser.add_argument("file_svd", type=str, help="Path to the SVD predictions CSV file.")
    parser.add_argument("file_cbf", type=str, help="Path to the CBF predictions CSV file.")
    
    args = parser.parse_args()
    
    calculate_rmse(args.file_svd, args.file_cbf)
