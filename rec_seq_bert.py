# -*- coding: utf-8 -*-
"""LLM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z0xw36QwD6kCAxLzfnJ3DXR2Cv-evNoW

# 大規模言語モデルのファインチューニング

# 1 環境の準備
"""
import torch
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pprint import pprint
from datasets import Dataset, ClassLabel
from datasets import load_dataset
import pandas as pd
from typing import Union
from transformers import BatchEncoding
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error

# 乱数シードを42に固定
set_seed(42)
print("乱数シード設定完了")

"""# 2 データセットの準備"""

# Hugging Face Hub上のllm-book/wrime-sentimentのリポジトリからデータを読み込む
# original_train_dataset = load_dataset("llm-book/wrime-sentiment", split="train")
# valid_dataset = load_dataset("llm-book/wrime-sentiment", split="validation")

# original_train_dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split="train")
# valid_dataset = load_dataset("shunk031/JGLUE", name="MARC-ja", split="validation")

# 学習データからN個のデータだけを抽出
# train_dataset = original_train_dataset.shuffle(seed=42).select([i for i in range(1000)])

# # CSVファイルからデータを読み込む
# original_train_df = pd.read_csv('/content/llm-class/dataset/train.csv')
# valid_df = pd.read_csv('/content/llm-class/dataset/validation.csv')
# train_dataset = Dataset.from_pandas(original_train_df)
# valid_dataset = Dataset.from_pandas(valid_df)


from sklearn.preprocessing import MinMaxScaler

# CSVファイルからデータを読み込む
original_train_df = pd.read_csv('rec-class/dataset/train_for_bert.csv')
valid_df = pd.read_csv('rec-class/dataset/validation_for_bert.csv')

# ラベルの正規化用にMinMaxScalerを作成
scaler = MinMaxScaler()

# trainデータのラベルを正規化
original_train_df['label'] = scaler.fit_transform(original_train_df[['label']])

# validデータのラベルを正規化
valid_df['label'] = scaler.transform(valid_df[['label']])

# 正規化後のデータをDatasetに変換
train_dataset = Dataset.from_pandas(original_train_df)
valid_dataset = Dataset.from_pandas(valid_df)


# pprintで見やすく表示する
pprint(train_dataset[0])

print("")

"""# 3. トークン化"""

# モデル名を指定してトークナイザを読み込む
# model_name = "cl-tohoku/bert-base-japanese-v3"
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# トークナイザのクラス名を確認
print(type(tokenizer).__name__)

# UserID、MovieIDを別れないようにトークンを登録する。
# user_tokens = [f"user_{i}" for i in range(1, 101)]
# tokenizer.add_tokens(user_tokens)
# movie_tokens = [f"movie_{i}" for i in range(1, 1001)]
# tokenizer.add_tokens(movie_tokens)
# rating_tokens = [str(round(x, 1)) for x in np.arange(1.0, 5.1, 0.1)]
# tokenizer.add_tokens(rating_tokens)


# テキストのトークン化
tokens = tokenizer.tokenize(train_dataset[0]['sentence'])
print(tokens)

# データのトークン化

def preprocess_text_classification(
    example: dict[str, str | int]
) -> BatchEncoding:
    """文書分類の事例のテキストをトークナイズし、IDに変換"""
    encoded_example = tokenizer(example["sentence"].replace("[SEP]",""), max_length=512)
    # 各IDがどのトークンを表すかを表示
    input_tokens = tokenizer.convert_ids_to_tokens(encoded_example["input_ids"])
    #print("Input Tokens:", input_tokens)
    # モデルの入力引数である"labels"をキーとして格納する
    encoded_example["labels"] = float(example["label"])  # ラベルをFloat型に変換
    return encoded_example


encoded_train_dataset = train_dataset.map(
    preprocess_text_classification,
    remove_columns=train_dataset.column_names,
)
encoded_valid_dataset = valid_dataset.map(
    preprocess_text_classification,
    remove_columns=valid_dataset.column_names,
)

# トークン化の確認
print(encoded_train_dataset[0])

"""# 4 ミニバッチ構築"""

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# ミニバッチ結果の確認
batch_inputs = data_collator(encoded_train_dataset[0:4])
pprint({name: tensor.size() for name, tensor in batch_inputs.items()})

"""# 5 モデルの準備"""

from transformers import AutoModelForSequenceClassification
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 1

model = (AutoModelForSequenceClassification
    .from_pretrained(model_name, num_labels=1)
    .to(device))


# モデルの埋め込み層にも新しいトークンを追加
model.resize_token_embeddings(len(tokenizer))



"""# 6 訓練の実行"""

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output_wrime",  # 結果の保存フォルダ
    per_device_train_batch_size=8,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=8,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type="linear",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=3,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="1/rmse",  # 最良のモデルを決定する評価指標
    fp16=False,  # 修正: FP16を無効にする
    fp16_full_eval=False,  # 修正: FP16 full evalを無効にする
)

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    rmse = mean_squared_error(labels, logits, squared=False)

    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = 1/rmse
    
    return {"mse": mse, "mae": mae, "r2": r2, "1/rmse": accuracy}

trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics_for_regression,
)
trainer.train()

latest_eval_metrics = trainer.evaluate()
print(latest_eval_metrics)

predictions = trainer.predict(encoded_valid_dataset)

# 通常は0番目のラベルに対応する予測値
predictions_df = pd.DataFrame({
    'label': valid_dataset["label"],
    'sentence': valid_dataset["sentence"],
    'predicted_value': predictions.predictions.flatten()
})

# MinMaxScalerで元のスケールに戻す
original_label = scaler.inverse_transform(predictions_df[['label']])
original_predicted_labels = scaler.inverse_transform(predictions_df[['predicted_value']])

# 通常は0番目のラベルに対応する予測値
predictions_df_2 = pd.DataFrame({
    'label': original_label.flatten(),
    'sentence': valid_dataset["sentence"],
    'predicted_value': original_predicted_labels.flatten()
})

# 予測結果をCSVに保存
predictions_df_2.to_csv("rec-class/dataset/results_lmm.csv", index=False)

mse_original_scale = mean_squared_error(original_label, original_predicted_labels)
mae_original_scale = mean_absolute_error(original_label, original_predicted_labels)
rmse_original_scale = np.sqrt(mse_original_scale)

# Display the results
print("MSE:", mse_original_scale)
print("MAE:", mae_original_scale)
print("RMSE:", rmse_original_scale)

# Save the best model
trainer.save_model("rec-class/best_model")

# Load the best model for prediction
best_model = AutoModelForSequenceClassification.from_pretrained("rec-class/best_model").to(device)

# Load the test dataset
test_df = pd.read_csv('rec-class/dataset/test_for_bert.csv')
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the test dataset
encoded_test_dataset = test_dataset.map(
    preprocess_text_classification,
    remove_columns=test_dataset.column_names,
)

# Make predictions on the test dataset using the best model
test_predictions = trainer.predict(encoded_test_dataset)

# Convert predictions to the original scale
original_test_predicted_labels = scaler.inverse_transform(test_predictions.predictions.flatten().reshape(-1, 1))


# Create a DataFrame with test predictions
submission_df = pd.DataFrame({
    'userId_movieId':test_df["sentence"],
    'rating': original_test_predicted_labels.flatten()
})

# 処理と新しい列の作成
submission_df["userId_movieId"] = submission_df["userId_movieId"].apply(lambda text: text.split("[SEP]")[0].split("_")[1].strip() + "_" + text.split("[SEP]")[1].split(" ")[-2].split("_")[1].strip())

# userId_movieIdごとにratingの平均値を計算
average_ratings = submission_df.groupby('userId_movieId')['rating'].mean().reset_index()

# 結果を保存
average_ratings.to_csv("rec-class/dataset/submission.csv", index=False)
print("提出用ファイル作成完了しました。submission.csvをダウンロードしてKaggleに登録ください。")


