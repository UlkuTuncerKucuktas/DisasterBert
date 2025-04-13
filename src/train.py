import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.dataset import DisasterDataset
from openpyxl import Workbook

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}") 
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

class ExcelLoggerCallback(TrainerCallback):
    def __init__(self, filename):
        self.filename = filename
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.append(["epoch", "global_step", "split", "accuracy", "precision", "recall", "f1"])
        self.wb.save(self.filename)
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        row = [state.epoch, state.global_step, "test", metrics.get("eval_accuracy", None), metrics.get("eval_precision", None), metrics.get("eval_recall", None), metrics.get("eval_f1", None)]
        self.ws.append(row)
        self.wb.save(self.filename)
        return control

def main():
    parser = argparse.ArgumentParser(description="Train a BERT model on disaster datasets.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name from HuggingFace Hub")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset splits (train, dev, test)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and evaluation batch size")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_name = os.path.basename(os.path.normpath(args.dataset_dir))
    train_path = os.path.join(args.dataset_dir, f"{base_name}_train.tsv")
    test_path = os.path.join(args.dataset_dir, f"{base_name}_test.tsv")

    train_dataset = DisasterDataset(data_path=train_path, tokenizer=tokenizer, max_len=350,desc_csv_path="/app/DisasterBert/desc.csv", prompt_heuristic=True)
    test_dataset = DisasterDataset(data_path=test_path, tokenizer=tokenizer, max_len=350, label_encoder=train_dataset.label_encoder,desc_csv_path="/app/DisasterBert/desc.csv", prompt_heuristic=True)

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(train_dataset.label_encoder.classes_))

    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        disable_tqdm=False,
        label_smoothing_factor=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
    )

    excel_filename = f"{base_name}_{args.model_name.replace('/', '-')}.xlsx"
    excel_logger = ExcelLoggerCallback(excel_filename)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[excel_logger]
    )

    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    excel_logger.ws.append([trainer.state.epoch, trainer.state.global_step, "test", test_results.get("eval_accuracy", None), test_results.get("eval_precision", None), test_results.get("eval_recall", None), test_results.get("eval_f1", None)])
    excel_logger.wb.save(excel_filename)

    print("\nFinal Test Metrics:")
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()
