# !/usr/bin/env python
# coding: utf-8

import json
import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics import f1_score, accuracy_score

from grokfast import gradfilter_ema
from eval_metrics import evaluate_rule_classification


# --------------------- Dataset ---------------------
class RuleQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, custom_tokens=True,
                 skip_numbers=False, rule_category=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.custom_tokens = custom_tokens
        self.skip_numbers = skip_numbers
        self.rule_category = rule_category

        # Load rule categories
        self.rule2cats = pd.read_json('../data/interim/classified_rules.json')[
            ['rule', 'cat-mapping']].set_index('rule')['cat-mapping'].to_dict()
        self.rule2cats['Safe'] = ['safe']
        for k, v in self.rule2cats.items():
            if not len(v):
                self.rule2cats[k] = ['other']

    def __len__(self):
        return len(self.data)

    def format_rule(self, rule_n, rule_text):
        """Format a single rule with optional category and number."""
        formatted = rule_text
        if self.rule_category:
            cats = ', '.join(self.rule2cats.get(rule_text, ['other']))
            formatted = f"{cats}: {formatted}"
        if not self.skip_numbers and rule_n is not None:
            formatted = f"{rule_n}. {formatted}"
        return formatted

    def data_to_qa(self, entry):
        """Convert entry to QA format."""
        # Format rules
        rules = "[BOR]\n" if self.custom_tokens else ""
        rule_boundaries = {}

        sorted_rules = sorted(entry['community']['rules'].items())
        for rule_n, rule_text in sorted_rules:
            formatted = self.format_rule(rule_n, rule_text)
            rule_boundaries[str(rule_n)] = (len(rules), len(rules) + len(formatted))
            rules += formatted + '\n'

        rules = rules[:-1]  # Remove final newline
        if self.custom_tokens:
            rules += "\n[EOR]"

        # Get answer boundaries
        answer_start, answer_end = rule_boundaries[str(entry['applied_rule_n'])]

        # Format question
        community_id = entry['community']['actor_id']
        question = f"[BOQ]\n{community_id}\n[EOQ]" if self.custom_tokens else community_id

        # Format context
        context = rules
        if self.custom_tokens:
            context += f"\n[BOC]\n{entry['content']}\n[EOC]"
        else:
            context += f"\n{entry['content']}"

        return {
            'question': question,
            'context': context,
            'answer_start': answer_start,
            'answer_end': answer_end,
            'answer_rule_n': entry['applied_rule_n'],
            'answer_text': entry['applied_rule_text'],
            'rule_boundaries': rule_boundaries,
            'removed': entry['removed']
        }

    def __getitem__(self, idx):
        entry = self.data[idx]
        qa = self.data_to_qa(entry)

        # Tokenize
        encoding = self.tokenizer(
            qa['question'],
            qa['context'],
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # Convert char positions to token positions
        offset = encoding['offset_mapping'][0]
        sequence_ids = encoding.sequence_ids(0)

        # Find context boundaries
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Map rule boundaries to tokens
        rule_token_boundaries = {}
        idx = context_start
        for rule_n, (start_char, end_char) in sorted(qa['rule_boundaries'].items(),
                                                     key=lambda x: x[1][0]):
            # Find start token
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_token = idx - 1

            # Find end token
            while idx <= context_end and offset[idx][0] <= end_char:
                idx += 1
            end_token = idx - 1

            rule_token_boundaries[rule_n] = (start_token, end_token)

        # Get answer token positions
        answer_rule_n = str(qa['answer_rule_n'])
        start_pos, end_pos = rule_token_boundaries.get(answer_rule_n, (-1, -1))

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': start_pos,
            'end_positions': end_pos,
            'rule_token_boundaries': rule_token_boundaries,
            'answer_rule_n': qa['answer_rule_n'],
            'removed': qa['removed']
        }


# --------------------- Collator ---------------------
def collate_fn(batch):
    """Custom collate function to handle rule_token_boundaries dict."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    start_positions = torch.tensor([item['start_positions'] for item in batch])
    end_positions = torch.tensor([item['end_positions'] for item in batch])
    answer_rule_ns = [item['answer_rule_n'] for item in batch]
    removed = [item['removed'] for item in batch]

    # Collect all unique rules across batch
    all_rules = set()
    for item in batch:
        all_rules.update(item['rule_token_boundaries'].keys())

    # Create tensor dict for rule boundaries
    rule_token_boundaries = {}
    for rule_n in all_rules:
        starts = []
        ends = []
        for item in batch:
            if rule_n in item['rule_token_boundaries']:
                s, e = item['rule_token_boundaries'][rule_n]
                starts.append(s)
                ends.append(e)
            else:
                starts.append(-1)
                ends.append(-1)
        rule_token_boundaries[rule_n] = (torch.tensor(starts), torch.tensor(ends))

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'rule_token_boundaries': rule_token_boundaries,
        'answer_rule_number': answer_rule_ns,
        'answer_removed': removed
    }


# --------------------- Prediction Extraction ---------------------
def extract_predictions(outputs, batch):
    """Extract rule predictions from model outputs."""
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    batch_size = start_logits.size(0)

    predicted_rules = np.zeros(batch_size, dtype=int)

    for i in range(batch_size):
        s = start_logits[i].argmax().item()
        e = end_logits[i].argmax().item()
        if e < s:
            e = s

        best_rule = 0
        best_coverage = 0.0
        matched = False

        for rule_n, (rule_starts, rule_ends) in batch['rule_token_boundaries'].items():
            rs = rule_starts[i].item()
            re = rule_ends[i].item()

            # Exact match
            if rs <= s < re and rs < e <= re:
                predicted_rules[i] = int(rule_n)
                matched = True
                break

            # Fallback: best overlap
            if re > rs:
                overlap_start = max(rs, s)
                overlap_end = min(re, e + 1)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / (re - rs)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_rule = int(rule_n)

        if not matched:
            predicted_rules[i] = best_rule

    return predicted_rules


# --------------------- Training ---------------------
def train_epoch(model, train_loader, optimizer, device, alpha=0.98, lamb=2.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    grads = None
    criterion = torch.nn.CrossEntropyLoss()

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = criterion(outputs['start_logits'], start_positions) + \
               criterion(outputs['end_logits'], end_positions)

        loss.backward()
        optimizer.step()

        # Apply grokfast
        grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_loader), grads


# --------------------- Evaluation ---------------------
def evaluate(model, data_loader, device):
    """Evaluate model on data."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = criterion(outputs['start_logits'], start_positions) + \
                   criterion(outputs['end_logits'], end_positions)
            total_loss += loss.item()

            preds = extract_predictions(outputs, batch)
            all_preds.extend(preds)
            all_labels.extend(batch['answer_rule_number'])

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'predictions': all_preds,
        'labels': all_labels
    }


# --------------------- Data Preparation ---------------------
def prepare_data(json_path, tokenizer, custom_tokens=True, skip_numbers=False,
                 rule_category=False, max_length=512):
    """Load and prepare dataset."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Ensure rules are sorted
    for entry in data:
        rules_dict = entry['community']['rules']
        entry['community']['rules'] = dict(sorted(rules_dict.items()))

    dataset = RuleQADataset(data, tokenizer, max_length, custom_tokens,
                            skip_numbers, rule_category)
    return dataset


# --------------------- Performance Analysis ---------------------
def save_results(data, predictions, output_dir, test_name):
    """Save predictions and compute metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Load rule categories
    rule2cats = pd.read_json('../data/interim/classified_rules.json')[
        ['rule', 'cat-mapping']].set_index('rule')['cat-mapping'].to_dict()
    rule2cats['Safe'] = ['safe']
    for k, v in rule2cats.items():
        if not len(v):
            rule2cats[k] = ['other']

    rows = []
    for i, entry in enumerate(data):
        pred_rule_n = int(predictions[i])
        true_rule_n = entry['applied_rule_n']
        rules = entry['community']['rules']

        pred_rule_text = rules.get(pred_rule_n, '')
        true_rule_text = entry['applied_rule_text']

        rows.append({
            'comment_id': entry['ap_id'],
            'true_safe': true_rule_text == 'Safe',
            'true_rule_text': true_rule_text,
            'true_rule_category': rule2cats.get(true_rule_text, ['unknown']),
            'true_rule_n': true_rule_n,
            'predicted_safe': pred_rule_text == 'Safe',
            'predicted_rule_text': pred_rule_text,
            'community': entry['community']['actor_id'],
            'predicted_rule_category': rule2cats.get(pred_rule_text, ['unknown']),
            'predicted_rule_n': pred_rule_n,
            'n_rule_options': len(rules)
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'predictions_{test_name}.csv'), index=False)

    results, macro_f1 = evaluate_rule_classification(df, output_dir=output_dir)
    return macro_f1


# --------------------- Main ---------------------
def main(
        split_n=0,
        batch_size=16,
        n_epochs=1,
        lr=2e-4,
        weight_decay=0.001,
        alpha=0.98,
        lamb=2.0,
        custom_tokens=True,
        skip_numbers=False,
        rule_category=False,
        output_dir='../results/modq_extract'
):
    """Main training loop."""
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    data_path = f'../data/interim/splits/nonbinary/{split_n}'
    print("Loading data...")

    # Initialize model and tokenizer
    model_name = "DeepPavlov/bert-base-cased-conversational"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='../models/huggingface')
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir='../models/huggingface')

    # Add special tokens
    if custom_tokens:
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]", "[BOQ]", "[EOQ]"]
        })
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    # Prepare datasets
    train_dataset = prepare_data(f'{data_path}/train.json', tokenizer, custom_tokens,
                                 skip_numbers, rule_category)
    dev_dataset = prepare_data(f'{data_path}/dev.json', tokenizer, custom_tokens,
                               skip_numbers, rule_category)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    grads = None

    print(f"\nTraining for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")

        # Train
        train_loss, grads = train_epoch(model, train_loader, optimizer, device, alpha, lamb)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate on dev
        dev_results = evaluate(model, dev_loader, device)
        print(f"Dev Loss: {dev_results['loss']:.4f}")
        print(f"Dev Accuracy: {dev_results['accuracy']:.4f}")
        print(f"Dev Macro F1: {dev_results['macro_f1']:.4f}")

        # Save best model
        if dev_results['macro_f1'] > best_f1:
            best_f1 = dev_results['macro_f1']
            best_epoch = epoch
            model_save_path = os.path.join(output_dir, 'best_model')
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"New best model saved! F1: {best_f1:.4f}")

    print(f"\n=== Training Complete ===")
    print(f"Best epoch: {best_epoch + 1}, Best F1: {best_f1:.4f}")

    # Evaluate on test sets
    print("\n=== Evaluating on test sets ===")
    test_sets = ['test_stratified', 'test_n_rules_out', 'test_n_communities_out']

    for test_name in test_sets:
        print(f"\nEvaluating on {test_name}...")
        with open(f'{data_path}/{test_name}.json', 'r') as f:
            test_data = json.load(f)

        test_dataset = prepare_data(f'{data_path}/{test_name}.json', tokenizer,
                                    custom_tokens, skip_numbers, rule_category)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn)

        test_results = evaluate(model, test_loader, device)
        print(f"{test_name} Macro F1: {test_results['macro_f1']:.4f}")

        # Save detailed results
        test_output_dir = os.path.join(output_dir, f'split_{split_n}', test_name)
        macro_f1 = save_results(test_data, test_results['predictions'],
                                test_output_dir, test_name)


if __name__ == '__main__':
    main(
        split_n=0,
        batch_size=64,
        n_epochs=1,
        lr=2e-4,
        weight_decay=0.001,
        alpha=0.98,
        lamb=2.0,
        custom_tokens=True,
        skip_numbers=False,
        rule_category=False,
        output_dir='../results/modq_extract'
    )