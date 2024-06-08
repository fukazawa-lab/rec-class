import argparse
import torch
import os
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging

# ロギングレベルを変更
logging.basicConfig(level=logging.ERROR)

def main(epoch_num, model_name, data_folder):
    # 乱数シードを42に固定
    set_seed(42)
    print("乱数シード設定完了")

    # CSVファイルからデータを読み込む
    original_train_df = pd.read_csv(data_folder+ 'training_bert.csv')
    valid_df = pd.read_csv(data_folder+ 'validation_bert.csv')

    # トークナイザを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # データのトークン化
    def preprocess_text_classification(example):
        encoded_example = tokenizer(example["sentence"], max_length=512)
        encoded_example["labels"] = float(example["label"])  # ラベルをFloat型に変換
        return encoded_example

    train_dataset = Dataset.from_pandas(original_train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    encoded_train_dataset = train_dataset.map(
        preprocess_text_classification,
        remove_columns=train_dataset.column_names,
    )
    encoded_valid_dataset = valid_dataset.map(
        preprocess_text_classification,
        remove_columns=valid_dataset.column_names,
    )

    # ミニバッチ構築用のデータコレーター
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # モデルの読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    # 訓練の実行
    training_args = TrainingArguments(
        output_dir="output_wrime",  # 結果の保存フォルダ
        per_device_train_batch_size=8,  # 訓練時のバッチサイズ
        per_device_eval_batch_size=8,  # 評価時のバッチサイズ
        learning_rate=2e-5,  # 学習率
        lr_scheduler_type="linear",  # 学習率スケジューラの種類
        warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
        num_train_epochs=epoch_num,  # エポック数
        save_strategy="epoch",  # チェックポイントの保存タイミング
        logging_strategy="epoch",  # ロギングのタイミング
        evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
        load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
        metric_for_best_model="1/mse",  # 最良のモデルを決定する評価指標
        fp16=False,  # 修正: FP16を無効にする
        fp16_full_eval=False,  # 修正: FP16 full evalを無効にする
        disable_tqdm=True,  # tqdmの進行状況バーを無効にする
    )

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

    # 予測結果をDataFrameに格納
    predictions_df = pd.DataFrame({
        'userId_movieId': valid_df["userId_itemId"],
        'label': valid_dataset["label"],
        'predicted_value': predictions.predictions.flatten()
    })

    # 提出用ファイル作成
    predictions_df.to_csv(data_folder+"validation_predictions_llm.csv", index=False)

    mse_original_scale = mean_squared_error(predictions_df['label'], predictions_df['predicted_value'])
    mae_original_scale = mean_absolute_error(predictions_df['label'], predictions_df['predicted_value'])
    rmse_original_scale = np.sqrt(mse_original_scale)

    # 結果表示
    print("MAE:", mae_original_scale)
    print("MSE:", mse_original_scale)
    print("RMSE:", rmse_original_scale)

    test_file=data_folder+ 'test_bert.csv'

    # テストデータの予測
    if test_file:
        test_df = pd.read_csv(test_file)

        test_dataset = Dataset.from_pandas(test_df)
        encoded_test_dataset = test_dataset.map(
            preprocess_text_classification,
            remove_columns=test_dataset.column_names,
        )

        test_predictions = trainer.predict(encoded_test_dataset)

        # テスト結果をDataFrameに格納
        test_predictions_df = pd.DataFrame({
            'userId_movieId': test_df["userId_movieId"],
            'rating': test_predictions.predictions.flatten()
        })

        # 提出用ファイル作成
        test_predictions_df.to_csv("submission_llm.csv", index=False)

        print("提出用ファイル作成完了しました。submission_llm.csvをダウンロードしてKaggleに登録ください.")


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2= r2_score(labels, logits)
    accuracy = 1 / mse
    return {"mse": mse, "mae": mae, "r2": r2, "1/mse": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument("--epoch_num", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name for training.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the training CSV file.")

    args = parser.parse_args()

    main(args.epoch_num, args.model_name, args.data_folder)
