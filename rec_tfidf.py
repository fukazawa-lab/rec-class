import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import argparse

def main(kmeans_flag, num_clusters):
    # Load train and validation data
    train_data = pd.read_csv('rec-class/dataset/training.csv')
    validation_data = pd.read_csv('rec-class/dataset/validation.csv')

    # Dummy encode 'userId' in train and validation data
    train_data = pd.get_dummies(train_data, columns=['userId'], prefix='user')
    validation_data = pd.get_dummies(validation_data, columns=['userId'], prefix='user')

    # メタデータの読み込み
    metadata = pd.read_csv('rec-class/dataset/metadata.csv')  # 実際のパスに置き換えてください

    # 訓練データと検証データをメタデータで結合
    train_data = pd.merge(train_data, metadata, on='movieId', how='left')
    validation_data = pd.merge(validation_data, metadata, on='movieId', how='left')

    # Apply TF-IDF transformation to 'movieText' in train and validation data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the max_features parameter
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_data['description'])
    tfidf_matrix_validation = tfidf_vectorizer.transform(validation_data['description'])

    # Concatenate dummy-encoded userId columns and TF-IDF-transformed movieText in train and validation data
    user_columns = [col for col in train_data.columns if col.startswith('user_')]
    user_embeddings_train = train_data[user_columns].values
    features_train = np.concatenate((user_embeddings_train, tfidf_matrix_train.toarray()), axis=1)

    user_embeddings_validation = validation_data[user_columns].values
    features_validation = np.concatenate((user_embeddings_validation, tfidf_matrix_validation.toarray()), axis=1)

    # Use KMeans for dimensionality reduction if kmeans_flag is True
    if kmeans_flag:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        features_train = kmeans.fit_transform(features_train)
        features_validation = kmeans.transform(features_validation)

    # Extract target variable (ratings)
    target_train = train_data['rating']
    target_validation = validation_data['rating']

    # Train a Random Forest regression model
    model = RandomForestRegressor()
    model.fit(features_train, target_train)

    # Make predictions on the validation set
    predictions_validation = model.predict(features_validation)

    # Evaluate the model on the validation set
    rmse_validation = sqrt(mean_squared_error(target_validation, predictions_validation))
    print(f'Validation RMSE: {rmse_validation}')

    # Load test data
    test_original_data = pd.read_csv('rec-class/dataset/test.csv')

    # Dummy encode 'userId' in test data
    test_data = pd.get_dummies(test_original_data, columns=['userId'], prefix='user')

    # 訓練データとメタデータを結合
    test_data = pd.merge(test_data, metadata, on='movieId', how='left')

    # Apply TF-IDF transformation to 'movieText' in test data
    tfidf_matrix_test = tfidf_vectorizer.transform(test_data['description'])

    # Concatenate dummy-encoded userId columns and TF-IDF-transformed movieText in test data
    user_embeddings_test = test_data[user_columns].values
    features_test = np.concatenate((user_embeddings_test, tfidf_matrix_test.toarray()), axis=1)

    # Use KMeans for dimensionality reduction on test data if kmeans_flag is True
    if kmeans_flag:
        features_test = kmeans.transform(features_test)

    # Make predictions on the test set
    predictions = model.predict(features_test)

    # Create a DataFrame with test data and predictions
    submission_df = test_original_data[['userId', 'movieId']].copy()
    submission_df['rating'] = predictions

    # userIdとmovieIdを結合して新しい列uid_iidを作成
    submission_df['userId_movieId'] = submission_df['userId'].astype(str) + '_' + submission_df['movieId'].astype(str)

    # 必要な列だけを抽出して出力
    output_data = submission_df[['userId_movieId', 'rating']]
    output_data.to_csv('rec-class/dataset/submission.csv', index=False)
    print("提出用ファイル作成完了しました。submission.csvをダウンロードしてKaggleに登録ください.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation system script.")
    parser.add_argument("--kmeans_flag", type=bool, default=False, help="Whether to apply KMeans for dimensionality reduction")
    parser.add_argument("--num_clusters", type=int, default=100, help="Number of clusters for KMeans")

    args = parser.parse_args()
    main(args.kmeans_flag, args.num_clusters)
