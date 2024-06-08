import pandas as pd
import argparse
import os

def create_bert_input_csv(file_name_in, file_name_out, sentence, data_folder):

    # CSVファイルを読み込みます
    df = pd.read_csv(file_name_in)
    user_profile_df = pd.read_csv(data_folder + "user_profile_data.csv")
    metadata_df = pd.read_csv(data_folder+ "metadata.csv")

    user_profile_df['userId'] = user_profile_df['userId'].astype(str)
    metadata_df['itemId'] = metadata_df['itemId'].astype(str)

    # user_profile_data.csvを辞書に変換します（userIdからprofileへのマッピング）
    user_profile_dict = user_profile_df.set_index('userId')['profile'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからtitleへのマッピング）
    item_title_dict = metadata_df.set_index('itemId')['title'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからcategoryへのマッピング）
    item_category_dict = metadata_df.set_index('itemId')['category'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからdescriptionへのマッピング）
    item_description_dict = metadata_df.set_index('itemId')['description'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからdescriptionへのマッピング）
    item_option1_dict = metadata_df.set_index('itemId')['option1'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからdescriptionへのマッピング）
    item_option2_dict = metadata_df.set_index('itemId')['option2'].to_dict()

    # metadata_all.csvを辞書に変換します（itemIdからdescriptionへのマッピング）
    item_option3_dict = metadata_df.set_index('itemId')['option3'].to_dict()
    

    # 結果を保存するリスト
    sentences = []
    labels = []
    pairs = []

    # CSVの各行に対して処理を行います
    for _, row in df.iterrows():
        user_id = str(row['userId'])
        item_id = str(row['itemId'])
        rating = row['rating']

        # user_profile_data.csvから該当するprofileを取得します
        profile = user_profile_dict.get(user_id, "")

        # metadata_all.csvから該当するtitleを取得します
        title = item_title_dict.get(item_id, "")

        # metadata_all.csvから該当するcategoryを取得します
        category = item_category_dict.get(item_id, "")

        # metadata_all.csvから該当するdescriptionを取得します
        description = item_description_dict.get(item_id, "")

        # metadata_all.csvから該当するdescriptionを取得します
        option1 = item_option1_dict.get(item_id, "")

        # metadata_all.csvから該当するdescriptionを取得します
        option2 = item_option2_dict.get(item_id, "")

        # metadata_all.csvから該当するdescriptionを取得します
        option3 = item_option3_dict.get(item_id, "")
        
        # 文をフォーマットします
        formatted_sentence = sentence.format( user_id=user_id, item_id=item_id,profile=profile, title=title, category=category, description=description, option1=option1, option2=option2, option3=option3)

        # リストに追加します
        sentences.append(formatted_sentence)
        labels.append(rating)
        pairs.append(f"{user_id}_{item_id}")

    # 新しいDataFrameに変換します
    bert_df = pd.DataFrame({
        'sentence': sentences,
        'label': labels,
        'userId_itemId': pairs
    })

    # 新しいCSVファイルに保存します
    bert_df.to_csv(file_name_out, index=False)

    print(f"BERT input CSV created and saved to {file_name_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BERT input CSV files.")
    parser.add_argument("--sentence", type=str, required=True, help="Template sentence for BERT input.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder.")

    args = parser.parse_args()

    for file_name in ["training", "validation", "test"]:
        history_file_in = os.path.join(args.data_folder, f"{file_name}.csv")
        history_file_out = os.path.join(args.data_folder, f"{file_name}_bert.csv")

        if os.path.exists(history_file_in):
            create_bert_input_csv(history_file_in, history_file_out, args.sentence, args.data_folder)
        else:
            print(f"{file_name}.csv が見つかりません。スキップします。")
