import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import argparse
import os

def main(kmeans_flag, num_clusters, datafolder):
    # 訓練データと検証データの読み込み
    train_data = pd.read_csv(datafolder+ "training.csv")
    validation_data = pd.read_csv( datafolder+ "validation.csv")

    # 'userId'列をstr型に変換
    train_data['userId'] = train_data['userId'].astype(str)
    validation_data['userId'] = validation_data['userId'].astype(str)

    # ユーザープロファイルデータの読み込み
    user_profiles = pd.read_csv( datafolder+ "user_profile_data.csv")
    user_profiles['userId'] = user_profiles['userId'].astype(str)

    # TF-IDFベクトル化
    profile_tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    profile_tfidf_matrix = profile_tfidf_vectorizer.fit_transform(user_profiles['profile'])

    # userIdをインデックスとしてユーザープロファイルTF-IDFベクトルを取得する辞書を作成
    user_profile_tfidf_dict = {user_profiles.iloc[i]['userId']: profile_tfidf_matrix[i].toarray() for i in range(len(user_profiles))}

    # メタデータの読み込み
    metadata = pd.read_csv(datafolder+ "metadata.csv")

    # descriptionカラムがNaNの場合、titleカラムの値をコピーする
    metadata['description'] = metadata['description'].fillna(metadata['title'])

    # 訓練データと検証データをメタデータで結合
    train_data = pd.merge(train_data, metadata, on='itemId', how='left')
    validation_data = pd.merge(validation_data, metadata, on='itemId', how='left')

    # 訓練データと検証データの'description'にTF-IDF変換を適用
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_data['description'])
    tfidf_matrix_validation = tfidf_vectorizer.transform(validation_data['description'])

    # 訓練データと検証データのユーザープロファイルTF-IDFベクトルを取得
    user_embeddings_train = np.array([user_profile_tfidf_dict[user_id][0] for user_id in train_data['userId']])
    user_embeddings_validation = np.array([user_profile_tfidf_dict[user_id][0] for user_id in validation_data['userId']])

    # features_trainの作成
    features_train = np.concatenate((user_embeddings_train, tfidf_matrix_train.toarray()), axis=1)
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
    validation_output = validation_data[['userId', 'itemId']].copy()
    validation_output['true_rating'] = target_validation
    validation_output['rating'] = predictions_validation

    # userIdとitemIdを結合して新しい列userId_itemIdを作成
    validation_output['userId_itemId'] = validation_output['userId'].astype(str) + '_' + validation_output['itemId'].astype(str)

    # 必要な列だけを抽出して出力
    validation_output_final = validation_output[['userId_itemId', 'true_rating', 'rating']]
    validation_output_final.to_csv(datafolder+ 'validation_predictions_tfidf.csv', index=False)
    print("検証結果をvalidation_predictions_tfidf.csvに保存しました。")

    test_file_path=datafolder+ 'test.csv'
    if not os.path.exists(test_file_path):
        # print("test.csv が見つからないため、テストデータの予測をスキップします。")
        return


    # テストデータの読み込み
    test_original_data = pd.read_csv(test_file_path)
    
    # テストデータをメタデータと結合
    test_data = pd.merge(test_original_data, metadata, on='itemId', how='left')

    # テストデータの'description'にTF-IDF変換を適用
    tfidf_matrix_test = tfidf_vectorizer.transform(test_data['description'])

    # テストデータのユーザープロファイルTF-IDFベクトルを取得
    # user_embeddings_test = np.array([user_profile_tfidf_dict[user_id][0] for user_id in test_data['userId']])
    user_embeddings_test = np.array([user_profile_tfidf_dict.get(user_id, profile_tfidf_vectorizer.transform([""]).toarray())[0] for user_id in test_data['userId']])

    # features_testの作成
    features_test = np.concatenate((user_embeddings_test, tfidf_matrix_test.toarray()), axis=1)

    # kmeans_flagがTrueの場合はテストデータに次元削減を適用
    if kmeans_flag:
        features_test = kmeans.transform(features_test)

    # テストセットでの予測
    predictions = model.predict(features_test)

    # テストデータと予測値を含むデータフレームを作成
    submission_df = test_original_data[['userId', 'itemId']].copy()
    submission_df['rating'] = predictions

    # userIdとitemIdを結合して新しい列userId_itemIdを作成
    submission_df['userId_movieId'] = submission_df['userId'].astype(str) + '_' + submission_df['itemId'].astype(str)

    # 必要な列だけを抽出して出力
    output_data = submission_df[['userId_movieId', 'rating']]
    output_data.to_csv(datafolder + 'submission_tfidf.csv', index=False)
    print(f"提出用ファイル作成完了しました。{datafolder}submission_tfidf.csvをダウンロードしてKaggleに登録ください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="レコメンデーションシステムのスクリプトを実行します。")
    parser.add_argument("--kmeans_flag", type=bool, default=False, help="次元削減のためにKMeansを適用するかどうか")
    parser.add_argument("--num_clusters", type=int, default=100, help="KMeansのクラスタ数")
    parser.add_argument("--datafolder", type=str, required=True, help="CSVファイルへのパス")

    args = parser.parse_args()
    main(args.kmeans_flag, args.num_clusters, args.datafolder)
