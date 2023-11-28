import pandas as pd
import random
import argparse

# 設定
SEP = "[SEP]"
MASK = "[MASK]"

def generate_data(infile, outfile, rep_num, seq_num):

    # validation.csvの読み込み
    df_valid = pd.read_csv('rec-class/dataset/training.csv')

    # userinfo.csvの読み込み
    df_userinfo = pd.read_csv('rec-class/dataset/userinfo.csv')

    # 処理結果を保存するデータフレーム
    result_rows = []

    # rep_num回水増しする
    for _ in range(rep_num):
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

                # 選択肢のリスト
                choices = [5, 6, 7]

                # ランダムに1つの数を選ぶ
                random_choice = random.choice(choices)

                # ランダムにk個の整数を選ぶ
                selected_indices = random.sample(range(threshold), seq_num)
                # sorted_indices = sorted(selected_indices)

                # 選ばれた整数に対応する要素を取り出してリストにする
                selected_movies = [movies[i] for i in selected_indices]
                selected_ratings = [ratings[i] for i in selected_indices]

                # 2つ目の"[SEP]"の前にmovieIdを入れる
                movie_id = f"movie_{str(int(row['movieId']))}"
                user_sequence = (
                    user_sequence + SEP + " " + ' '.join(selected_movies) +
                    " " + movie_id + " " + SEP + " " + ' '.join(selected_ratings) + " " + MASK
                )

                # labelにratingを追加
                label = row['rating']

                # 結果を保存
                result_rows.append({'sentence': user_sequence, 'label': label})

    # 結果をデータフレームに変換
    result_df = pd.DataFrame(result_rows)

    # 結果をCSVファイルに書き出し
    result_df.to_csv('rec-class/dataset/train_for_bert.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented data for training a BERT model.")
    parser.add_argument("--rep_num", type=int, help="Number of repetitions", required=True)
    parser.add_argument("--seq_num", type=int, help="Number of selected elements in each sequence", required=True)
    parser.add_argument("--infile", type=str, default="rec-class/dataset/validation.csv", help="Input CSV file path")
    parser.add_argument("--outfile", type=str, default="rec-class/dataset/validation_for_bert.csv", help="Output CSV file path")

    # コマンドライン引数をパース
    args = parser.parse_args()

    # メイン関数の呼び出し
    generate_data(args.infile, args.outfile, args.rep_num, args.seq_num)
