import pandas as pd
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SlopeOne, CoClustering
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

def train_algo(algo_name,datafolder, epochs, n_factors, lr):
    # SurpriseライブラリのReaderを定義（評価値の範囲は1から5まで）
    reader = Reader(rating_scale=(1, 5))

    # 訓練データと検証データの読み込み
    training_data = pd.read_csv(datafolder+ "training.csv")
    validation_data = pd.read_csv( datafolder+ "validation.csv")

    # モデルのトレーニング
    training_data = Dataset.load_from_df(training_data[['userId', 'itemId', 'rating']], reader)
    trainset = training_data.build_full_trainset()

    model = get_algorithm(algo_name, n_epochs=epochs, n_factors=n_factors, lr=lr)
    model.fit(trainset)

    # 検証データに対する予測
    validation_data = Dataset.load_from_df(validation_data[['userId', 'itemId', 'rating']], reader)
    validation_set = validation_data.build_full_trainset().build_testset()
    validation_predictions = model.test(validation_set)

    # RMSE, MAE, MSEの計算
    true_ratings = [prediction.r_ui for prediction in validation_predictions]
    predicted_ratings = [prediction.est for prediction in validation_predictions]
    
    validation_rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
    validation_mae = mean_absolute_error(true_ratings, predicted_ratings)
    validation_mse = mean_squared_error(true_ratings, predicted_ratings)
    
    print(f"MAE: {validation_mae}")
    print(f"MSE: {validation_mse}")
    print(f"RMSE: {validation_rmse}")

    # 検証データの予測結果をDataFrameに変換
    validation_df = pd.DataFrame(validation_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    validation_df = validation_df.rename(columns={'uid': 'userId', 'iid': 'itemId', 'r_ui': 'true_rating', 'est': 'rating'})
    validation_df['userId_movieId'] = validation_df['userId'].astype(str) + '_' + validation_df['itemId'].astype(str)
    validation_output = validation_df[['userId_movieId', 'true_rating', 'rating']]
    validation_output.to_csv(datafolder+f'validation_predictions_{algo_name}.csv', index=False)
    print(f"検証データの予測結果をvalidation_predictions_{algo_name}.csvに保存しました。")

    test_file_path=datafolder+ 'test.csv'
    if not os.path.exists(test_file_path):
        # print("test.csv が見つからないため、テストデータの予測をスキップします。")
        return

    # テストデータの読み込み
    test_data = pd.read_csv(test_file_path)

    # テストデータに対する予測
    test_data = Dataset.load_from_df(test_data[['userId', 'itemId', 'rating']], reader)
    test_set = test_data.build_full_trainset().build_testset()
    test_predictions = model.test(test_set)

    # テストデータの予測結果をDataFrameに変換
    test_df = pd.DataFrame(test_predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    test_df = test_df.rename(columns={'uid': 'userId', 'iid': 'movieId', 'r_ui': 'true_rating', 'est': 'rating'})
    test_df['userId_movieId'] = test_df['userId'].astype(str) + '_' + test_df['movieId'].astype(str)
    test_output = test_df[['userId_movieId', 'rating']]
    test_output.to_csv(datafolder + f'submission_{algo_name}.csv', index=False)
    print(f"提出用ファイル作成完了しました。submission_{algo_name}.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation system script.")
    parser.add_argument("--algo", type=str, default='SVD', help="Algorithm to use for training (e.g., SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SlopeOne, CoClustering)")
    parser.add_argument("--datafolder", type=str, required=True, help="CSVファイルへのパス")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--n_factors", type=int, default=100, help="Number of factors for SVD")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for SVD")

    args = parser.parse_args()
    train_algo(args.algo, args.datafolder, args.epochs, args.n_factors, args.lr)
