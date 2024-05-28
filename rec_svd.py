import pandas as pd
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SlopeOne, CoClustering
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse
import os


def get_algorithm(algo_name, **kwargs):
    if algo_name == 'SVD':
        return SVD(n_epochs=kwargs.get('n_epochs', 30), n_factors=kwargs.get('n_factors', 10), lr_all=kwargs.get('lr', 0.005))
    elif algo_name == 'SVDpp':
        return SVDpp(n_epochs=kwargs.get('n_epochs', 30), n_factors=kwargs.get('n_factors', 100), lr_all=kwargs.get('lr', 0.005))
    elif algo_name == 'NMF':
        return NMF(n_epochs=kwargs.get('n_epochs', 30), n_factors=kwargs.get('n_factors', 100))
    elif algo_name == 'KNNBasic':
        return KNNBasic()
    elif algo_name == 'KNNWithMeans':
        return KNNWithMeans()
    elif algo_name == 'KNNWithZScore':
        return KNNWithZScore()
    elif algo_name == 'KNNBaseline':
        return KNNBaseline()
    elif algo_name == 'SlopeOne':
        return SlopeOne()
    elif algo_name == 'CoClustering':
        return CoClustering()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def train_algo(algo_name, training_file, validation_file, test_file, epochs, n_factors, lr):
    # SurpriseライブラリのReaderを定義（評価値の範囲は1から5まで）
    reader = Reader(rating_scale=(1, 5))

    # データの読み込み
    training_data = pd.read_csv(training_file)
    validation_data = pd.read_csv(validation_file)

    # モデルのトレーニング
    training_data = Dataset.load_from_df(training_data[['userId', 'movieId', 'rating']], reader)
    trainset = training_data.build_full_trainset()

    model = get_algorithm(algo_name, n_epochs=epochs, n_factors=n_factors, lr=lr)
    model.fit(trainset)

    # 検証データに対する予測
    validation_data = Dataset.load_from_df(validation_data[['userId', 'movieId', 'rating']], reader)
    validation_set = validation_data.build_full_trainset().build_testset()
    validation_predictions = model.test(validation_set)

    # RMSEの計算
    validation_rmse = sqrt(mean_squared_error([prediction.est for prediction in validation_predictions], [prediction.r_ui for prediction in validation_predictions]))
    print(f"Validation RMSE: {validation_rmse}")

    # 検証データの予測結果をDataFrameに変換
    validation_df = pd.DataFrame(validation_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    validation_df = validation_df.rename(columns={'uid': 'userId', 'iid': 'movieId', 'r_ui': 'true_rating', 'est': 'rating'})
    validation_df['userId_movieId'] = validation_df['userId'].astype(str) + '_' + validation_df['movieId'].astype(str)
    validation_output = validation_df[['userId_movieId', 'true_rating', 'rating']]
    validation_output.to_csv(f'rec-class/dataset/validation_predictions_{algo_name}.csv', index=False)
    print(f"検証データの予測結果をvalidation_predictions_{algo_name}.csvに保存しました。")

    """# 7 テストデータの予測"""
    test_file_path = '/content/rec-class/dataset/test.csv'
    if not os.path.exists(test_file_path):
        print("test.csv が見つからないため、テストデータの予測をスキップします。")
        return

    # テストデータの読み込み
    test_data = pd.read_csv(test_file)


    # テストデータに対する予測
    test_data = Dataset.load_from_df(test_data[['userId', 'movieId', 'rating']], reader)
    test_set = test_data.build_full_trainset().build_testset()
    test_predictions = model.test(test_set)

    # テストデータの予測結果をDataFrameに変換
    test_df = pd.DataFrame(test_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    test_df = test_df.rename(columns={'uid': 'userId', 'iid': 'movieId', 'r_ui': 'true_rating', 'est': 'rating'})
    test_df['userId_movieId'] = test_df['userId'].astype(str) + '_' + test_df['movieId'].astype(str)
    test_output = test_df[['userId_movieId', 'rating']]
    test_output.to_csv(f'rec-class/dataset/submission_{algo_name}.csv', index=False)
    print(f"提出用ファイル作成完了しました。submission_{algo_name}.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation system script.")
    parser.add_argument("--algo", type=str, default='SVD', help="Algorithm to use for training (e.g., SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SlopeOne, CoClustering)")
    parser.add_argument("--training_file", type=str, default='rec-class/dataset/training.csv', help="Path to the training data file")
    parser.add_argument("--validation_file", type=str, default='rec-class/dataset/validation.csv', help="Path to the validation data file")
    parser.add_argument("--test_file", type=str, default='rec-class/dataset/test.csv', help="Path to the test data file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--n_factors", type=int, default=100, help="Number of factors for SVD")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for SVD")

    args = parser.parse_args()
    train_algo(args.algo, args.training_file, args.validation_file, args.test_file, args.epochs, args.n_factors, args.lr)
