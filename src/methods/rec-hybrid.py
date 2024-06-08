import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

def calculate_rmse(file_svd, file_cbf):
    # CSVファイルの読み込み
    validation_predictions_svd = pd.read_csv(file_svd)
    validation_predictions_cbf = pd.read_csv(file_cbf)

    validation_predictions_cbf.rename(columns={'rating': 'rating_cbf'}, inplace=True)
    validation_predictions_svd.rename(columns={'rating': 'rating_svd'}, inplace=True)

    # userId_movieIdでデータを突き合わせる
    merged_df = pd.merge(validation_predictions_svd, validation_predictions_cbf, on='userId_itemId')

    # true_ratingのカラムが存在するかチェックし、存在しない場合はtrue_rating_xを使用
    if 'true_rating' in merged_df.columns:
        true_ratings = merged_df['true_rating']
    elif 'true_rating_x' in merged_df.columns:
        true_ratings = merged_df['true_rating_x']
    else:
        raise KeyError("Neither 'true_rating' nor 'true_rating_x' columns are present in the merged dataframe.")

    # αを0から1まで0.1刻みで変化させる
    alphas = np.arange(0, 1.1, 0.1)

    # RMSE, MAE, MSEを格納するリスト
    rmses = []
    maes = []
    mses = []

    # 各αに対して重みづけ和を計算し、RMSEを算出
    for alpha in alphas:
        # 重みづけ和の計算
        weighted_predictions = alpha * merged_df['rating_cbf'] + (1 - alpha) * merged_df['rating_svd']

        # RMSE, MAE, MSEの計算
        rmse = np.sqrt(mean_squared_error(true_ratings, weighted_predictions))
        mae = mean_absolute_error(true_ratings, weighted_predictions)
        mse = mean_squared_error(true_ratings, weighted_predictions)
        
        rmses.append(rmse)
        maes.append(mae)
        mses.append(mse)

        print(f"Alpha: {alpha:.1f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RMSE for different alpha values.")
    parser.add_argument("file_svd", type=str, help="Path to the SVD predictions CSV file.")
    parser.add_argument("file_cbf", type=str, help="Path to the CBF predictions CSV file.")
    
    args = parser.parse_args()
    
    calculate_rmse(args.file_svd, args.file_cbf)
