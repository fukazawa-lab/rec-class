import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os

def calculate_rmse(file_svd, file_cbf, datafolder):
    # CSVファイルの読み込み
    validation_predictions_svd = pd.read_csv(file_svd)
    validation_predictions_cbf = pd.read_csv(file_cbf)

    validation_predictions_cbf.rename(columns={'rating': 'rating_cbf'}, inplace=True)
    validation_predictions_svd.rename(columns={'rating': 'rating_svd'}, inplace=True)

    # userId_itemIdでデータを突き合わせる
    merged_df = pd.merge(validation_predictions_svd, validation_predictions_cbf, on='userId_itemId')

    # true_ratingのカラムを確認
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

    # 最適αを探す
    best_alpha = 0
    best_rmse = float('inf')

    for alpha in alphas:
        weighted_predictions = alpha * merged_df['rating_cbf'] + (1 - alpha) * merged_df['rating_svd']
        rmse = np.sqrt(mean_squared_error(true_ratings, weighted_predictions))
        mae = mean_absolute_error(true_ratings, weighted_predictions)
        mse = mean_squared_error(true_ratings, weighted_predictions)
        rmses.append(rmse)

        print(f"Alpha: {alpha:.1f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha:.1f} with RMSE: {best_rmse:.4f}")

    # --- ここからテストデータを使ったハイブリッド予測 ---
    test_file_path = os.path.join(datafolder, 'test.csv')
    if not os.path.exists(test_file_path):
        print(datafolder)
        # print("test.csv が見つからないため、テストデータの予測をスキップします。")
        return

    # 入力ファイル名からsubmission用ファイル名を作成
    submission_svd_file = file_svd.replace('test_predictions_', 'submission_')
    submission_cbf_file = file_cbf.replace('test_predictions_', 'submission_')

    if not os.path.exists(submission_svd_file) or not os.path.exists(submission_cbf_file):
        print("submission用ファイルが見つかりません。")
        return

    # submission CSVの読み込み
    submission_svd = pd.read_csv(submission_svd_file)
    submission_cbf = pd.read_csv(submission_cbf_file)

    submission_cbf.rename(columns={'rating': 'rating_cbf'}, inplace=True)
    submission_svd.rename(columns={'rating': 'rating_svd'}, inplace=True)

    # userId_itemIdで結合
    submission_merged = pd.merge(submission_svd, submission_cbf, on='userId_itemId')

    # 最適αでハイブリッド予測
    submission_merged['rating'] = best_alpha * submission_merged['rating_cbf'] + (1 - best_alpha) * submission_merged['rating_svd']

    # 不要カラム削除
    submission_final = submission_merged[['userId_itemId', 'rating']]

    # 保存
    submission_final.to_csv('submission_hybrid.csv', index=False)
    print("submission_hybrid.csv を保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RMSE for different alpha values and create hybrid submission.")
    parser.add_argument("file_svd", type=str, help="Path to the SVD predictions CSV file.")
    parser.add_argument("file_cbf", type=str, help="Path to the CBF predictions CSV file.")
    parser.add_argument("--datafolder", type=str, default="./", help="Folder path where test.csv is located.")
    
    args = parser.parse_args()
    
    calculate_rmse(args.file_svd, args.file_cbf, args.datafolder)
