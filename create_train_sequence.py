import pandas as pd
import random
import argparse

# 設定
SEP = "[SEP]"
MASK = "[MASK]"

def generate_data(infile, outfile, rep_num, seq_num, order):

    # validation.csvの読み込み
    df_valid = pd.read_csv(infile)

    # metadata.csvの読み込み
    metadata = pd.read_csv('rec-class/dataset/metadata.csv')

    # userinfo.csvの読み込み
    df_userinfo = pd.read_csv('rec-class/dataset/userinfo.csv')

    # 処理結果を保存するデータフレーム
    result_rows = []

    # rep_num回水増しする
    if seq_num != -1:
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

                  # ランダムにk個の整数を選ぶ
                  selected_indices = random.sample(range(threshold), seq_num)
                  
                  # 選ばれたインデックスを小さい順に並べ替える
                  if order == "sequential":
                      selected_indices = sorted(selected_indices)

                  # 選ばれた整数に対応する要素を取り出してリストにする
                  selected_movies = [movies[i] for i in selected_indices]
                  selected_ratings = [ratings[i] for i in selected_indices]

                  # movieIdに対応するdescriptionを取得し、連結する
                  selected_descriptions = []
                  for movie_id in selected_movies:
                      description = metadata[metadata['movieId'] == int(movie_id.split("_")[1])]['description'].values[0]
                      selected_descriptions.append(description)

                  # descriptionを"."で連結
                  movie_description_sequence = ". ".join(selected_descriptions)

                  # 2つ目の"[SEP]"の前にmovieIdを入れる
                  movie_id = f"movie_{str(int(row['movieId']))}"
                  target_movie_description = metadata[metadata['movieId'] == int(row['movieId'])]['description'].values[0]

                  user_sequence = (
                      user_sequence + SEP + " " + ' '.join(selected_movies) +
                      " " + movie_id + " " + SEP + " " + ' '.join(selected_ratings) + " " + MASK +
                      SEP +movie_description_sequence  
                  )
                  # labelにratingを追加
                  label = row['rating']

                  # 結果を保存
                  result_rows.append({'sentence': user_sequence, 'label': label})

    else:
        for index, row in df_userinfo.iterrows():
            user_id = row['user'] # ユーザーIDを整数として処理し、小数点以下があれば削除

            user_info = df_userinfo[df_userinfo['user'] == user_id]

            user_sentence = user_info['sentence'].values[0]

            # Iterate through each movie and create a sentence
            # sentenceを"[SEP]"で分割
            user_sequence, movie_sequence, rating_sequence = user_sentence.split(SEP)
            movies = movie_sequence.split()
            ratings = rating_sequence.split()

            for i in range(len(movies)):
              # Create a copy of the ratings list to avoid modifying the original list
              masked_ratings = ratings.copy()
              # labelにratingを追加
              label = masked_ratings[i]

              # Add [MASK] to the last element of the ratings corresponding to the current movie
              masked_ratings[i] = "[MASK]"

              # Create the sentence for the current movie
              sentence = user_sequence + SEP + " " + ' '.join(movies) + " " + SEP + " " + ' '.join(masked_ratings)

              # 結果を保存
              result_rows.append({'sentence': sentence, 'label': label})

    # 結果をデータフレームに変換
    result_df = pd.DataFrame(result_rows)

    # 結果をCSVファイルに書き出し
    result_df.to_csv(outfile, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented data for training a BERT model.")
    parser.add_argument("--rep_num", type=int, help="Number of repetitions", required=True)
    parser.add_argument("--seq_num", type=int, help="Number of selected elements in each sequence. if -1 we set max len", required=True)
    parser.add_argument("--order", type=str,  default="random", help="Order of each sequence (sequential/random)")
    parser.add_argument("--infile", type=str, default="rec-class/dataset/training.csv", help="Input CSV file path")
    parser.add_argument("--outfile", type=str, default="rec-class/dataset/train_for_bert.csv", help="Output CSV file path")

    # コマンドライン引数をパース
    args = parser.parse_args()

    # メイン関数の呼び出し
    generate_data(args.infile, args.outfile, args.rep_num, args.seq_num, args.order)
