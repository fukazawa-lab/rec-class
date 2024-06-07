import pandas as pd
from collections import Counter
import argparse
import os

def generate_user_profiles(data_folder, cat_freq):
    # ファイルパスの設定
    trainingfile = os.path.join(data_folder, 'training.csv')
    metadatafile = os.path.join(data_folder, 'metadata.csv')
    userprofilefile = os.path.join(data_folder, 'user_profile_data.csv')

    # データを読み込みます
    training_data = pd.read_csv(trainingfile)
    metadata = pd.read_csv(metadatafile)

    # movieIdを展開してユーザごとにgenreを取得します
    user_genres = []
    for user_movies in training_data.groupby('userId')['itemId']:
        genres = []
        for movie_id in user_movies[1]:
            genres.extend(metadata[metadata['itemId'] == movie_id]['category'].astype(str))
        user_genres.append(', '.join(genres))

    # ユーザのDataFrameを作成します
    user_df = pd.DataFrame({'userId': training_data['userId'].unique(), 'profile': user_genres})

    # 結果を保存するリスト
    user_profiles = []
    profile_created_count = 0

    # ユーザごとに処理を行います
    for _, row in user_df.iterrows():
        user_id = row['userId']
        profile_text = row['profile']

        try:
            # プロファイルテキストをコンマで分割してリストに変換します
            profile_list = profile_text.split(', ')

            # 頻度をカウントします
            name_counter = Counter(profile_list)

            # 頻度がcat_freq回以上のものを抽出します
            frequent_names = [name for name, count in name_counter.items() if count >= cat_freq]

            # ユニークな要素に変換します
            unique_names = list(set(frequent_names))

            # コンマで連結します
            profile_summary = ', '.join(unique_names)

            if profile_summary:
                profile_summary = f"This person likes shopping genres such as {profile_summary}."
                profile_created_count += 1
            else:
                profile_summary = "No valid profile information available."
        except (ValueError, SyntaxError, TypeError):
            # エラーが発生した場合、プロファイルサマリを空にします
            profile_summary = "No valid profile information available."

        # 結果を保存します
        user_profiles.append({
            'userId': user_id,
            'profile': profile_summary
        })

    # 新しいDataFrameに変換します
    user_profile_df = pd.DataFrame(user_profiles)
    user_profile_df.to_csv(userprofilefile, index=False)

    # プロファイルが作成されたユーザ数と全ユーザ数を出力
    total_users = len(user_df)
    print(f"Total users: {total_users}")
    print(f"Users with created profiles: {profile_created_count}")

    return user_profile_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ユーザープロファイルを生成します。")
    parser.add_argument("--data_folder", type=str, required=True, help="データフォルダのパス")
    parser.add_argument("--cat_freq", type=int, required=True, help="頻度の閾値")

    args = parser.parse_args()
    user_profile_df = generate_user_profiles(args.data_folder, args.cat_freq)
