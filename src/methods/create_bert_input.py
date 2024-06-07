import pandas as pd
import argparse
import os

def create_bert_input_csv(file_name_in, file_name_out, sentence, user_profile_file, metadata_file):
    # CSVファイルを読み込みます
    df = pd.read_csv(file_name_in)
    user_profile_df = pd.read_csv(user_profile_file)
    metadata_df = pd.read_csv(metadata_file)

    # user_profile_data.csvを辞書に変換します（userIdからprofileへのマッピング）
    user_profile_dict = user_profile_df.set_index('userId')['profile'].to_dict()

    # metadata_all.csvを辞書に変換します（movieIdからtitleへのマッピング）
    movie_title_dict = metadata_df.set_index('movieId')['title'].to_dict()

    # metadata_all.csvを辞書に変換します（movieIdからgenreへのマッピング）
    movie_genre_dict = metadata_df.set_index('movieId')['genres'].to_dict()

    # metadata_all.csvを辞書に変換します（movieIdからoverviewへのマッピング）
    movie_overview_dict = metadata_df.set_index('movieId')['overview'].to_dict()

    # metadata_all.csvを辞書に変換します（movieIdからoverviewへのマッピング）
    movie_release_date_dict = metadata_df.set_index('movieId')['release_date'].to_dict()

    # metadata_all.csvを辞書に変換します（movieIdからoverviewへのマッピング）
    movie_runtime_dict = metadata_df.set_index('movieId')['runtime'].to_dict()


    # 結果を保存するリスト
    sentences = []
    labels = []
    pairs = []

    # CSVの各行に対して処理を行います
    for _, row in df.iterrows():
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = row['rating']

        # user_profile_data.csvから該当するprofileを取得します
        profile = user_profile_dict.get(user_id, "")

        # metadata_all.csvから該当するtitleを取得します
        title = movie_title_dict.get(movie_id, "")

        # metadata_all.csvから該当するgenreを取得します
        genre = movie_genre_dict.get(movie_id, "")

        # metadata_all.csvから該当するoverviewを取得します
        overview = movie_overview_dict.get(movie_id, "")

        # metadata_all.csvから該当するoverviewを取得します
        release_date = movie_release_date_dict.get(movie_id, "")

        # metadata_all.csvから該当するoverviewを取得します
        runtime = movie_runtime_dict.get(movie_id, "")

        # 文をフォーマットします
        formatted_sentence = sentence.format( user_id=user_id, movie_id=movie_id,profile=profile, title=title, genre=genre, overview=overview, release_date=release_date, runtime=runtime)

        # リストに追加します
        sentences.append(formatted_sentence)
        labels.append(rating)
        pairs.append(f"{user_id}_{movie_id}")

    # 新しいDataFrameに変換します
    bert_df = pd.DataFrame({
        'sentence': sentences,
        'label': labels,
        'userId_movieId': pairs
    })

    # 新しいCSVファイルに保存します
    bert_df.to_csv(file_name_out, index=False)

    print(f"BERT input CSV created and saved to {file_name_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BERT input CSV files.")
    parser.add_argument("--sentence", type=str, required=True, help="Template sentence for BERT input.")
    parser.add_argument("--user_profile_file", type=str, required=True, help="Path to the user profile data CSV file.")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--history_file_in", type=str, required=True, help="Path to the input training CSV file.")
    parser.add_argument("--history_file_out", type=str, required=True, help="Path to save the output training BERT CSV file.")

    args = parser.parse_args()

    # 関数を呼び出して指定されたファイルを処理します
    create_bert_input_csv(args.history_file_in, args.history_file_out, args.sentence, args.user_profile_file, args.metadata_file)

    if args.history_file_in is not None and args.history_file_out is not None:
        create_bert_input_csv(args.history_file_in, args.history_file_out, args.sentence, args.user_profile_file, args.metadata_file)
    else:
        print("test.csv が見つからないため、テストデータのBERT用ファイル作成をスキップします。")
