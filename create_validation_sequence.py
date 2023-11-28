import pandas as pd
import random
import argparse

# 設定
SEP = "[SEP]"
MASK = "[MASK]"

def generate_data(infile, outfile, rep_num, seq_num):
    # validation.csvの読み込み
    df_valid = pd.read_csv(infile)

    # userinfo.csvの読み込み
    df_userinfo = pd.read_csv('rec-class/dataset/userinfo.csv')

    # 処理結果を保存するデータフレーム
    result_rows = []

    for _ in range(rep_num):
        # validation.csvの行ごとに処理
        for index, row in df_valid.iterrows():
            user_id = f"user_{str(int(row['userId']))}"  # ユーザーIDを整数として処理し、小数点以下があれば削除
            user_info = df_userinfo[df_userinfo['user'] == user_id]

            # user_infoが存在する場合のみ処理を行う
            if not user_info.empty:
                user_sentence = user_info['sentence'].values[0]

                # sentenceを"[SEP]"で分割
                user_sequence, movie_sequence, rating_sequence = user_sentence.split(SEP)
                movies = movie_sequence.split()
                ratings = rating_sequence.split()
                threshold = len(movies)
                k_num = int(threshold * 0.6)

                # ランダムにk個の整数を選ぶ
                selected_indices = random.sample(range(threshold), seq_num)

                # 選ばれた整数に対応する要素を取り出してリストにする
                selected_movies = [movies[i] for i in selected_indices]
                selected_ratings = [ratings[i] for i in selected_indices]

                # 2つ目の"[SEP]"の前にmovieIdを入れる
                movie_id = f"movie_{str(int(row['movieId']))}"
                user_sequence = (
                    user_sequence + SEP + " " + ' '.join(selected_movies) + " " +
                    movie_id + " " + SEP + " " + ' '.join(selected_ratings) + " " + MASK
                )

                # labelにratingを追加
                label = row['rating']

                # 結果を保存
                result_rows.append({'sentence': user_sequence, 'label': label})

    # 結果をデータフレームに変換
    result_df = pd.DataFrame(result_rows)

    # 結果をCSVファイルに書き出し
    result_df.to_csv(outfile, index=False)

if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Generate data for BERT.")

    # コマンドライン引数の定義
    parser.add_argument("--rep_num", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--seq_num", type=int, default=4, help="Number of sequences")
    parser.add_argument("--infile", type=str, default="rec-class/dataset/validation.csv", help="Input CSV file path")
    parser.add_argument("--outfile", type=str, default="rec-class/dataset/validation_for_bert.csv", help="Output CSV file path")

    # コマンドライン引数をパース
    args = parser.parse_args()

    # メイン関数の呼び出し
    generate_data(args.infile, args.outfile, args.rep_num, args.seq_num)

# import pandas as pd

# import random
# # 設定
# SEP = "[SEP]"
# MASK = "[MASK]"

# # validation.csvの読み込み
# df_valid = pd.read_csv('rec-class/dataset/validation.csv')

# # userinfo.csvの読み込み
# df_userinfo = pd.read_csv('rec-class/dataset/userinfo.csv')

# # 処理結果を保存するデータフレーム
# result_rows = []

# # validation.csvの行ごとに処理
# for index, row in df_valid.iterrows():
#     user_id = f"user_{str(int(row['userId']))}"  # ユーザーIDを整数として処理し、小数点以下があれば削除
#     user_info = df_userinfo[df_userinfo['user'] == user_id]

#     # user_infoが存在する場合のみ処理を行う
#     if not user_info.empty:
#         user_sentence = user_info['sentence'].values[0]

#         # sentenceを"[SEP]"で分割
#         user_sequence, movie_sequence,rating_sequence = user_sentence.split(SEP)
#         movies = movie_sequence.split()
#         ratings = rating_sequence.split()

#         # 2つ目の"[SEP]"の前にmovieIdを入れる
#         movie_id = f"movie_{str(int(row['movieId']))}"

#         # Create the sentence for the current movie
#         sentence = user_sequence + SEP + " " + ' '.join(movies) +" "+ movie_id + " "+ SEP + " " + ' '.join(ratings) + " "+ MASK

#         # Print or store the resulting sentence for the current movie
#         print(sentence)

#         # labelにratingを追加
#         label = row['rating']

#         # 結果を保存
#         result_rows.append({'sentence': sentence, 'label': label})


# # 結果をデータフレームに変換
# result_df = pd.DataFrame(result_rows)

# # 結果をCSVファイルに書き出し
# result_df.to_csv('rec-class/dataset/validation_for_bert.csv', index=False)

