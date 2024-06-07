import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import argparse
import os

def main(kmeans_flag, num_clusters):
    # 訓練データと検証データの読み込み
    train_data = pd.read_csv('rec-class/dataset/training.csv')
    validation_data = pd.read_csv('rec-class/dataset/validation.csv')

    # 訓練データと検証データの'userId'をダミーエンコード
    train_data = pd.get_dummies(train_data, columns=['userId'], prefix='user')
    validation_data = pd.get_dummies(validation_data, columns=['userId'], prefix='user')

    # メタデータの読み込み
    metadata = pd.read_csv('rec-class/dataset/metadata.csv')  # 実際のパスに置き換えてください

    # 訓練データと検証データをメタデータで結合
    train_data = pd.merge(train_data, metadata, on='movieId', how='left')
    validation_data = pd.merge(validation_data, metadata, on='movieId', how='left')

    # 訓練データと検証データの'movieText'にTF-IDF変換を適用
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # max_featuresパラメータを調整できます
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_data['overview'])
    tfidf_matrix_validation = tfidf_vectorizer.transform(validation_data['overview'])

    # 訓練データと検証データにダミーエンコードされた'userId'列とTF-IDF変換された'movieText'を結合
    user_columns = [col for col in train_data.columns if col.startswith('user_')]
    user_embeddings_train = train_data[user_columns].values
    features_train = np.concatenate((user_embeddings_train, tfidf_matrix_train.toarray()), axis=1)

    user_embeddings_validation = validation_data[user_columns].values
    features_validation = np.concatenate((user_embeddings_validation, tfidf_matrix_validation.toarray()), axis=1)

    # kmeans_flagがTrueの場合は次元削減のためにKMeansを使用
    if kmeans_flag:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        features_train = kmeans.fit_transform(features_train)
        features_validation = kmeans.transform(features_validation)

    # 目的変数（レーティング）を抽出
    target_train = train_data['rating']
    target_validation = validation_data['rating']

    # ランダムフォレスト回帰モデルの訓練
    model = RandomForestRegressor()
    model.fit(features_train, target_train)

    # 検証セットでの予測
    predictions_validation = model.predict(features_validation)

    # 検証セットでモデルを評価
    rmse_validation = sqrt(mean_squared_error(target_validation, predictions_validation))
    mae_validation = mean_absolute_error(target_validation, predictions_validation)
    mse_validation = mean_squared_error(target_validation, predictions_validation)
    
    print(f'MAE: {mae_validation}')
    print(f'MSE: {mse_validation}')
    print(f'RMSE: {rmse_validation}')

    # predictions_validationをCSVに出力
    validation_output = pd.read_csv('rec-class/dataset/validation.csv')[['userId', 'movieId']].copy()
    validation_output['true_rating'] = target_validation
    validation_output['rating'] = predictions_validation

    # userIdとmovieIdを結合して新しい列userId_movieIdを作成
    validation_output['userId_movieId'] = validation_output['userId'].astype(str) + '_' + validation_output['movieId'].astype(str)

    # 必要な列だけを抽出して出力
    validation_output_final = validation_output[['userId_movieId', 'true_rating', 'rating']]
    validation_output_final.to_csv('rec-class/dataset/validation_predictions_tfidf.csv', index=False)
    print("検証結果をvalidation_predictions_tfidf.csvに保存しました。")

    """# 7 テストデータの予測"""
    test_file_path = '/content/rec-class/dataset/test.csv'
    if not os.path.exists(test_file_path):
        print("test.csv が見つからないため、テストデータの予測をスキップします。")
        return

    # テストデータの読み込み
    test_original_data = pd.read_csv(test_file_path)
    
    # テストデータの'userId'をダミーエンコード
    test_data = pd.get_dummies(test_original_data, columns=['userId'], prefix='user')

    # テストデータをメタデータと結合
    test_data = pd.merge(test_data, metadata, on='movieId', how='left')

    # テストデータの'movieText'にTF-IDF変換を適用
    tfidf_matrix_test = tfidf_vectorizer.transform(test_data['overview'])

    # テストデータにダミーエンコードされた'userId'列とTF-IDF変換された'movieText'を結合
    user_embeddings_test = test_data[user_columns].values
    features_test = np.concatenate((user_embeddings_test, tfidf_matrix_test.toarray()), axis=1)

    # kmeans_flagがTrueの場合はテストデータに次元削減を適用
    if kmeans_flag:
        features_test = kmeans.transform(features_test)

    # テストセットでの予測
    predictions = model.predict(features_test)

    # テストデータと予測値を含むデータフレームを作成
    submission_df = test_original_data[['userId', 'movieId']].copy()
    submission_df['rating'] = predictions

    # userIdとmovieIdを結合して新しい列userId_movieIdを作成
    submission_df['userId_movieId'] = submission_df['userId'].astype(str) + '_' + submission_df['movieId'].astype(str)

    # 必要な列だけを抽出して出力
    output_data = submission_df[['userId_movieId', 'rating']]
    output_data.to_csv('rec-class/dataset/submission_tfidf.csv', index=False)
    print("提出用ファイル作成完了しました。submission_tfidf.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="レコメンデーションシステムのスクリプトを実行します。")
    parser.add_argument("--kmeans_flag", type=bool, default=False, help="次元削減のためにKMeansを適用するかどうか")
    parser.add_argument("--num_clusters", type=int, default=100, help="KMeansのクラスタ数")

    args = parser.parse_args()
    main(args.kmeans_flag, args.num_clusters)
