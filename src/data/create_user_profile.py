import pandas as pd
from collections import Counter
import argparse
import os

def generate_user_profiles(data_folder, cat_freq):
    trainingfile = os.path.join(data_folder, 'training.csv')
    metadatafile = os.path.join(data_folder, 'metadata.csv')
    userprofilefile = os.path.join(data_folder, 'user_profile_data.csv')

    training_data = pd.read_csv(trainingfile)
    metadata = pd.read_csv(metadatafile)

    # itemId -> category を辞書化して高速化（重要）
    item2cat = metadata.set_index('itemId')['category'].to_dict()

    user_genres = []

    for user_id, user_movies in training_data.groupby('userId')['itemId']:
        genres = []

        for movie_id in user_movies:
            if movie_id in item2cat:
                cat_str = str(item2cat[movie_id])

                # ★ここが修正ポイント（カンマ区切りを分解）
                split_cats = [c.strip() for c in cat_str.split(',') if c.strip() != ""]

                genres.extend(split_cats)

        user_genres.append(', '.join(genres))

    user_df = pd.DataFrame({
        'userId': training_data['userId'].unique(),
        'profile': user_genres
    })

    user_profiles = []
    profile_created_count = 0

    for _, row in user_df.iterrows():
        user_id = row['userId']
        profile_text = row['profile']

        try:
            profile_list = [p.strip() for p in profile_text.split(',') if p.strip() != ""]

            name_counter = Counter(profile_list)

            frequent_names = [
                name for name, count in name_counter.items()
                if count >= cat_freq
            ]

            unique_names = list(set(frequent_names))

            profile_summary = ', '.join(unique_names)

            if profile_summary:
                profile_summary = f"This person likes genres such as {profile_summary}."
                profile_created_count += 1
            else:
                profile_summary = "No valid profile information available."

        except Exception:
            profile_summary = "No valid profile information available."

        user_profiles.append({
            'userId': user_id,
            'profile': profile_summary
        })

    user_profile_df = pd.DataFrame(user_profiles)
    user_profile_df.to_csv(userprofilefile, index=False)

    total_users = len(user_df)
    print(f"Total users: {total_users}")
    print(f"Users with created profiles: {profile_created_count}")

    return user_profile_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ユーザープロファイルを生成します。")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--cat_freq", type=int, required=True)

    args = parser.parse_args()
    user_profile_df = generate_user_profiles(args.data_folder, args.cat_freq)
