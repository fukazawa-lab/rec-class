import pandas as pd
import argparse

def create_hybrid_submission(file_svd, file_cbf, alpha):
    # CSVファイルの読み込み
    submission_svd = pd.read_csv(file_svd)
    submission_cbf = pd.read_csv(file_cbf)

    submission_cbf.rename(columns={'rating': 'rating_cbf'}, inplace=True)
    submission_svd.rename(columns={'rating': 'rating_svd'}, inplace=True)

    # userId_movieIdでデータを突き合わせる
    merged_test_df = pd.merge(submission_cbf, submission_svd, on='userId_movieId')

    # 重み付け和の計算
    weighted_predictions = alpha * merged_test_df['rating_cbf'] + (1 - alpha) * merged_test_df['rating_svd']

    # 結果をDataFrameにまとめる
    submission_hybrid = pd.DataFrame({
        'userId_movieId': merged_test_df['userId_movieId'],
        'rating': weighted_predictions
    })

    # submission_hybrid.csvとして保存
    submission_hybrid.to_csv('submission_hybrid.csv', index=False)

    print("提出用ファイル作成完了しました。submission_hybrid.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create hybrid submission file.")
    parser.add_argument("file_svd", type=str, help="Path to the SVD submission CSV file.")
    parser.add_argument("file_cbf", type=str, help="Path to the CBF submission CSV file.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CBF predictions (default: 0.5)")

    args = parser.parse_args()
    create_hybrid_submission(args.file_svd, args.file_cbf, args.alpha)
