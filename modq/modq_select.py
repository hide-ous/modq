import json
import os
import torch

import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForMultipleChoice, Trainer,
    TrainingArguments, TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score

# --------------------- Dataset ---------------------
class RuleQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example["content"] + " [SEP] " + example["community"]["name"]
        choices = list(example["community"]["rules"].values())

        if example['applied_rule_text'] not in choices:
            print(f"[!] Rule text not found in choices for idx={idx}")
            print("Rule Text:", example['applied_rule_text'])
            print("Choices:", choices)
            raise ValueError("Label rule text not found in rule choices.")

        label = choices.index(example['applied_rule_text'])

        encodings = self.tokenizer(
            [context] * len(choices),
            choices,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": label,
            "num_choices": len(choices)
        }

# --------------------- Collator ---------------------
class VariableChoiceCollator:
    def __init__(self, max_num_choices):
        self.max_num_choices = max_num_choices

    def __call__(self, features):
        max_choices = self.max_num_choices
        max_len = features[0]["input_ids"].shape[1]

        def pad_tensor(tensor, target_shape):
            pad_size = (target_shape[0] - tensor.shape[0], 0)
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size[0]), value=0)

        input_ids = torch.stack([pad_tensor(f["input_ids"], (max_choices, max_len)) for f in features])
        attention_mask = torch.stack([pad_tensor(f["attention_mask"], (max_choices, max_len)) for f in features])
        labels = torch.tensor([f["label"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --------------------- Metrics ---------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average='macro')
    }

# --------------------- Best Model Saver ---------------------
class BestF1SaverCallback(TrainerCallback):
    def __init__(self):
        self.best_f1 = 0.0
        self.best_model_path = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_macro_f1" not in metrics:
            return
        macro_f1 = metrics["eval_macro_f1"]
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            print(f"New best dev macro F1: {macro_f1:.4f}")
            best_model_dir = os.path.join(args.output_dir, "best_model")
            kwargs["model"].save_pretrained(best_model_dir)
            kwargs["model"].config.save_pretrained(best_model_dir)
            self.best_model_path = best_model_dir

# --------------------- Utility ---------------------
def get_max_num_choices(datasets):
    return max(len(example["community"]["rules"]) for dataset in datasets for example in dataset)

def evaluate_and_save(trainer, data, name, tokenizer, output_dir, interim_path):
    dataset = RuleQADataset(data, tokenizer)
    output = trainer.predict(dataset)
    preds = output.predictions.argmax(-1)
    labels = output.label_ids

    # Load category mapping
    rule2cats = pd.read_json(f'{interim_path}/classified_rules.json')[['rule', 'cat-mapping']].set_index('rule')['cat-mapping'].to_dict()
    rule2cats['Safe'] = ['safe']
    for k, v in rule2cats.items():
        if not len(v):
            rule2cats[k] = ['other']

    rows = []
    for i, ex in enumerate(data):
        all_choices = list(ex["community"]["rules"].values())
        true_rule_text = ex["applied_rule_text"]
        pred_rule_text = all_choices[preds[i]]

        rows.append({
            "comment_id": ex["ap_id"],
            "true_safe": true_rule_text == "Safe",
            "true_rule_text": true_rule_text,
            "true_rule_category": str(rule2cats.get(true_rule_text, ['unknown'])),
            "true_rule_n": all_choices.index(true_rule_text),
            "predicted_safe": pred_rule_text == "Safe",
            "predicted_rule_text": pred_rule_text,
            "community": ex["community"]["name"],
            "predicted_rule_category": str(rule2cats.get(pred_rule_text, ['unknown'])),
            "predicted_rule_n": preds[i],
            "n_rule_options": len(all_choices),
        })

    df = pd.DataFrame(rows)
    output_csv = os.path.join(output_dir, f"predictions_{name}.csv")
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved predictions to {output_csv}")

    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"{name} Macro F1: {macro_f1:.4f}")
    return macro_f1

# --------------------- Main ---------------------
if __name__ == '__main__':
    interim_path = '../data/interim/'
    split_path = f'{interim_path}/splits/nonbinary/0'

    train_data = json.load(open(f'{split_path}/train.json'))[:500]
    eval_data = json.load(open(f'{split_path}/dev.json'))[:500]

    # Load test sets
    test_sets = {
        "test_stratified": json.load(open(f'{split_path}/test_stratified.json')),
        "test_n_rules_out": json.load(open(f'{split_path}/test_n_rules_out.json')),
        "test_n_communities_out": json.load(open(f'{split_path}/test_n_communities_out.json'))
    }

    max_num_choices = get_max_num_choices([train_data, eval_data])
    model_name = "DeepPavlov/bert-base-cased-conversational"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)

    train_dataset = RuleQADataset(train_data, tokenizer)
    eval_dataset = RuleQADataset(eval_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="../results/modq_select",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        fp16=True,
        logging_dir="../logs",
        logging_steps=10,
    )

    data_collator = VariableChoiceCollator(max_num_choices=max_num_choices)
    f1_callback = BestF1SaverCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[f1_callback]
    )

    trainer.train()

    # Load and evaluate best model
    best_model_path = os.path.join(training_args.output_dir, "best_model")
    best_model = AutoModelForMultipleChoice.from_pretrained(best_model_path).to(device)
    trainer.model = best_model

    for name, test_data in test_sets.items():
        evaluate_and_save(trainer, test_data, name, tokenizer, training_args.output_dir, interim_path)
