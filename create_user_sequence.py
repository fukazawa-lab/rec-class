import pandas as pd
from sklearn.model_selection import train_test_split

# 訓練データの読み込み
df = pd.read_csv('rec-class/dataset/training.csv')

# [SEP]と[MASK]を定義
SEP = "[SEP]"
MASK = "[MASK]"
CLS="[CLS]"


# カスタム関数でsentenceとlabelを作成

def create_sentence_label(group):
    user_id = f"user_{str(group['userId'].iloc[0])}"
    movie_ids = [f"movie_{str(i)}" for i in group['movieId']]
    ratings = group['rating'].astype(str).tolist()
    sequence_length=len(movie_ids)

    # sequence_lengthごとに分割
    movie_sequences = [movie_ids[i:i + sequence_length] for i in range(0, len(movie_ids), sequence_length)]
    rating_sequences = [ratings[i:i + sequence_length] for i in range(0, len(ratings), sequence_length)]

    # 各シーケンスに対してsentenceとlabelを作成
    result_rows = []
    for i in range(len(movie_sequences)):
        seq = movie_sequences[i]
        rating = rating_sequences[i]

        # [SEP]トークンを使って履歴と評価を結合し、[MASK]トークンを挿入
        label = user_id
        sentence = f" {user_id} {SEP} {' '.join([f'{movie}' for movie in seq])} {SEP} {' '.join([f'{r}' for r in rating])}"
        result_rows.append({'sentence': sentence, 'user': label})
    return pd.DataFrame(result_rows)

# グループごとにapply
result_df = df.groupby('userId').apply(lambda x: create_sentence_label(x)).reset_index(drop=True)

# 新しいCSVファイルに保存
result_df.to_csv('rec-class/dataset/userinfo.csv', index=False)




# import pandas as pd

# import random
# # 設定
# SEP = "[SEP]"
# MASK = "[MASK]"

# # validation.csvの読み込み
# df_valid = pd.read_csv('rec-class/dataset/training.csv')

# # userinfo.csvの読み込み
# df_userinfo = pd.read_csv('rec-class/dataset/userinfo.csv')

# # 処理結果を保存するデータフレーム
# result_rows = []

# # 3回水増しする
# for _ in range(1):
#     for index, row in df_userinfo.iterrows():
#         user_id = row['user'] # ユーザーIDを整数として処理し、小数点以下があれば削除
#         print(user_id)
#         user_info = df_userinfo[df_userinfo['user'] == user_id]

#         user_sentence = user_info['sentence'].values[0]

#         # Iterate through each movie and create a sentence
#         # sentenceを"[SEP]"で分割
#         user_sequence, movie_sequence, rating_sequence = user_sentence.split(SEP)
#         movies = movie_sequence.split()
#         ratings = rating_sequence.split()
#         for i in range(len(movies)):


#             # Create a copy of the ratings list to avoid modifying the original list
#             masked_ratings = ratings.copy()

#             # labelにratingを追加
#             label = masked_ratings[i]

#             # Add [MASK] to the last element of the ratings corresponding to the current movie
#             masked_ratings[i] = "[MASK]"

#             # Create the sentence for the current movie
#             sentence = user_sequence + SEP + " " + ' '.join(movies) + " " + SEP + " " + ' '.join(masked_ratings)

#             # Print or store the resulting sentence for the current movie
#             print(sentence)

#             # 結果を保存
#             result_rows.append({'sentence': sentence, 'label': label})

# # 結果をデータフレームに変換
# result_df = pd.DataFrame(result_rows)

# # 結果をCSVファイルに書き出し
# result_df.to_csv('rec-class/dataset/train_for_bert.csv', index=False)

