import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse

def train_svd(training_file, validation_file, test_file, epochs, n_factors, lr):
    # SurpriseライブラリのReaderを定義（評価値の範囲は1から5まで）
    reader = Reader(rating_scale=(1, 5))

    # データの読み込み
    training_data = pd.read_csv(training_file)
    validation_data = pd.read_csv(validation_file)

    # モデルのトレーニング
    training_data = Dataset.load_from_df(training_data[['userId', 'movieId', 'rating']], reader)
    trainset = training_data.build_full_trainset()

    model = SVD(n_epochs=epochs, n_factors=n_factors, lr_all=lr)
    model.fit(trainset)

    # 検証データに対する予測
    validation_data = Dataset.load_from_df(validation_data[['userId', 'movieId', 'rating']], reader)
    validation_set = validation_data.build_full_trainset().build_testset()
    predictions = model.test(validation_set)

    # RMSEの計算
    rmse = sqrt(mean_squared_error([prediction.est for prediction in predictions], [prediction.r_ui for prediction in predictions]))

    # 結果の表示
    print(f"Validation RMSE: {rmse}")

    # テストデータに対する予測
    test_data = pd.read_csv(test_file)
    test_data = Dataset.load_from_df(test_data[['userId', 'movieId', 'rating']], reader)
    test_set = test_data.build_full_trainset().build_testset()
    predictions = model.test(test_set)

    # 予測結果を DataFrame に変換
    submission_df = pd.DataFrame(predictions, columns=['userId', 'movieId', 'true_rating', 'rating', 'details'])
    submission_df = submission_df[['userId', 'movieId', 'rating']]

    # userIdとmovieIdを結合して新しい列uid_iidを作成
    submission_df['userId_movieId'] = submission_df['userId'].astype(str) + '_' + submission_df['movieId'].astype(str)

    # 必要な列だけを抽出して出力
    output_data = submission_df[['userId_movieId', 'rating']]
    output_data.to_csv('rec-class/dataset/submission.csv', index=False)
    print("提出用ファイル作成完了しました。submission.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation system script.")
    parser.add_argument("--training_file", type=str, default='rec-class/dataset/training.csv', help="Path to the training data file")
    parser.add_argument("--validation_file", type=str, default='rec-class/dataset/validation.csv', help="Path to the validation data file")
    parser.add_argument("--test_file", type=str, default='rec-class/dataset/test.csv', help="Path to the test data file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--n_factors", type=int, default=100, help="Number of factors for SVD")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for SVD")

    args = parser.parse_args()
    train_svd(args.training_file, args.validation_file, args.test_file, args.epochs, args.n_factors, args.lr)
