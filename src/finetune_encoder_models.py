#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tunes BERT-based or encoder models for sentiment classification with two approaches:
1. Full Fine-Tuning using MeanPooling and Accelerate.
2. Frozen Model with Classifier Attachment.

Includes hyperparameter optimization with Optuna, and logs best model configurations,
metrics, and hyperparameters in a .tsv file.

Batch size is included in the hyperparameter space, along with max_length.

Usage:
    python finetune_encoder_models.py --config_file configs/config.yaml
"""

import os
import random
import argparse
import logging
import yaml
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator,
)
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import optuna
from utils import load_sentiment_data
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Disable parallelism in tokenizers to avoid deadlocks in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Path for saving model results
RESULTS_FILE = None  # Will be set in main() based on model_name

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT-based or encoder models for sentiment classification.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()

def load_config(config_file: str):
    """Load configurations from a YAML file."""
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )

def preprocess_function(examples, tokenizer, max_length):
    """Tokenize and encode the dataset examples."""
    encoding = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )
    encoding['labels'] = examples['label']
    return encoding

class MeanPooling(torch.nn.Module):
    """Mean Pooling for generating sentence representation by averaging token embeddings."""
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

def save_results_to_file(results, RESULTS_FILE):
    """Save the model results to a TSV file."""
    df = pd.DataFrame(results)
    if not os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, sep='\t', index=False)
    else:
        df.to_csv(RESULTS_FILE, sep='\t', mode='a', header=False, index=False)

def train_full_finetune(configs, train_dataset, test_dataset, num_labels, results, dataset_name):
    """Train the model using full fine-tuning with Optuna for hyperparameter optimization and MeanPooling."""

    def objective(trial):
        # Suggest hyperparameters
        max_length = configs.get('max_length', 256)  # You can make this tunable with Optuna
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        num_epochs = trial.suggest_int('num_epochs', 2, 10)
        batch_size = configs.get('batch_size', 16)  # You can make this tunable with Optuna

        logging.info(f"Trial {trial.number}: max_length={max_length}, learning_rate={learning_rate}, num_epochs={num_epochs}, batch_size={batch_size}")

        # Update tokenizer
        tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])

        # Preprocess data with new max_length
        logging.info("Tokenizing and processing training data")
        train_dataset_processed = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_length),
            batched=True,
            desc="Tokenizing training data"
        )
        if test_dataset:
            logging.info("Tokenizing and processing test data")
            test_dataset_processed = test_dataset.map(
                lambda examples: preprocess_function(examples, tokenizer, max_length),
                batched=True,
                desc="Tokenizing test data"
            )

        # Initialize model, classifier, pooling layer, and accelerator
        model = AutoModel.from_pretrained(configs["model_name"])
        classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
        mean_pooling = MeanPooling()
        accelerator = Accelerator()

        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        # Prepare data loaders
        train_dataloader = DataLoader(train_dataset_processed, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
        if test_dataset:
            eval_dataloader = DataLoader(test_dataset_processed, batch_size=batch_size, collate_fn=default_data_collator)

        # Prepare everything with accelerator
        model, classifier, optimizer, train_dataloader = accelerator.prepare(model, classifier, optimizer, train_dataloader)
        if test_dataset:
            eval_dataloader = accelerator.prepare(eval_dataloader)

        # Training loop with tqdm progress bar
        model.train()
        classifier.train()
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
            for batch in progress_bar:
                optimizer.zero_grad()
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                pooled_output = mean_pooling(outputs, batch['attention_mask'])
                logits = classifier(pooled_output)
                loss = torch.nn.CrossEntropyLoss()(logits, batch['labels'])
                accelerator.backward(loss)
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())

        # Evaluation
        if test_dataset:
            model.eval()
            classifier.eval()
            all_predictions, all_labels = [], []
            eval_loss = 0
            progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False)
            with torch.no_grad():
                for batch in progress_bar:
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    pooled_output = mean_pooling(outputs, batch['attention_mask'])
                    logits = classifier(pooled_output)
                    loss = torch.nn.CrossEntropyLoss()(logits, batch['labels'])
                    eval_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    progress_bar.set_postfix(loss=loss.item())

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(all_labels, all_predictions),
                "f1": f1_score(all_labels, all_predictions, average='weighted'),
                "precision": precision_score(all_labels, all_predictions, average='weighted'),
                "recall": recall_score(all_labels, all_predictions, average='weighted'),
                "mcc": matthews_corrcoef(all_labels, all_predictions),
                "eval_loss": eval_loss / len(eval_dataloader),
            }
            logging.info(f"Evaluation Metrics: {metrics}")

            # Save best result
            result = {
                "model_type": "Full Fine-Tune",
                "hyperparameters": {
                    "max_length": max_length,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                },
                "metrics": metrics,
                "dataset": dataset_name,
                "model_name": configs["model_name"],
            }
            results.append(result)
            return -metrics["f1"]
        else:
            return 0  # No evaluation dataset available

    # Create Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    logging.info(f"Best hyperparameters for fine-tuning on {dataset_name}: {study.best_params}")

def train_frozen_with_classifier(configs, texts_train, labels_train, texts_test, labels_test, results, dataset_name):
    """Train a frozen encoder model with a classifier attached, optimized with Optuna."""
    classifiers = {
        'SVM': SVC,
        'Logistic Regression': LogisticRegression,
        'MLP': MLPClassifier,
        'LightGBM': LGBMClassifier
    }

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    model = AutoModel.from_pretrained(configs["model_name"])
    model.eval()  # Set model to evaluation mode

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize texts with max_length as a hyperparameter
    def objective(trial):
        max_length = configs.get('max_length', 256)  # You can make this tunable with Optuna
        batch_size = configs.get('batch_size', 8)  # You can make this tunable with Optuna

        logging.info(f"Trial {trial.number}: max_length={max_length}, batch_size={batch_size}")

        # Tokenize texts
        logging.info(f"Tokenizing texts with max_length={max_length}")
        train_encodings = tokenizer(
            texts_train,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )
        test_encodings = tokenizer(
            texts_test,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )

        # Create datasets
        train_dataset = Dataset.from_dict({**train_encodings, 'labels': labels_train})
        test_dataset = Dataset.from_dict({**test_encodings, 'labels': labels_test})

        # Tokenization progress bar
        train_dataset = train_dataset.map(
            lambda x: x,
            desc="Processing training dataset",
        )
        test_dataset = test_dataset.map(
            lambda x: x,
            desc="Processing test dataset",
        )

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_data_collator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_data_collator)

        # Get embeddings from the frozen model
        logging.info("Generating embeddings from the frozen model")
        mean_pooling = MeanPooling()
        train_embeddings = []
        train_labels_collected = []
        test_embeddings = []
        test_labels_collected = []

        # Generate train embeddings
        for batch in tqdm(train_dataloader, desc="Generating train embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                embeddings = mean_pooling(outputs, batch['attention_mask']).cpu().numpy()
                train_embeddings.append(embeddings)
                train_labels_collected.extend(batch['labels'].cpu().numpy())
        train_embeddings = np.vstack(train_embeddings)
        train_labels = np.array(train_labels_collected)

        # Generate test embeddings
        for batch in tqdm(test_dataloader, desc="Generating test embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                embeddings = mean_pooling(outputs, batch['attention_mask']).cpu().numpy()
                test_embeddings.append(embeddings)
                test_labels_collected.extend(batch['labels'].cpu().numpy())
        test_embeddings = np.vstack(test_embeddings)
        test_labels = np.array(test_labels_collected)

        # Standardize features
        logging.info("Standardizing features")
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        test_embeddings = scaler.transform(test_embeddings)

        # For each classifier, perform hyperparameter optimization and training
        best_f1 = -np.inf
        best_classifier = None
        best_metrics = None
        best_params = None

        for clf_name, clf_class in classifiers.items():
            logging.info(f"Training classifier: {clf_name}")

            def clf_objective(trial):
                # Suggest hyperparameters based on classifier type
                if clf_name == 'SVM':
                    params = {
                        'C': trial.suggest_loguniform('C', 1e-3, 1e2),
                        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                        'probability': True,
                    }
                elif clf_name == 'Logistic Regression':
                    params = {
                        'C': trial.suggest_loguniform('C', 1e-3, 1e2),
                        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                        'max_iter': 1000,
                    }
                elif clf_name == 'MLP':
                    params = {
                        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
                        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e-1),
                        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-3),
                        'max_iter': 500,
                    }
                elif clf_name == 'LightGBM':
                    params = {
                        'num_leaves': trial.suggest_int('num_leaves', 31, 70, step=10),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
                    }

                logging.info(f"Classifier Trial {trial.number}: {params}")

                classifier = clf_class(**params)
                classifier.fit(train_embeddings, train_labels)
                predictions = classifier.predict(test_embeddings)
                metrics = {
                    'accuracy': accuracy_score(test_labels, predictions),
                    'f1': f1_score(test_labels, predictions, average='weighted'),
                    'precision': precision_score(test_labels, predictions, average='weighted'),
                    'recall': recall_score(test_labels, predictions, average='weighted'),
                    'mcc': matthews_corrcoef(test_labels, predictions)
                }
                logging.info(f"{clf_name} Evaluation Metrics: {metrics}")

                nonlocal best_f1, best_classifier, best_metrics, best_params
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_classifier = clf_name
                    best_metrics = metrics
                    best_params = params

                return -metrics['f1']

            # Create Optuna study and optimize for the classifier
            classifier_study = optuna.create_study(direction='minimize')
            classifier_study.optimize(clf_objective, n_trials=5)
            logging.info(f"Best hyperparameters for {clf_name} on {dataset_name}: {classifier_study.best_params}")

        # After trying all classifiers, save the best one
        result = {
            "model_type": best_classifier,
            "hyperparameters": {
                "max_length": max_length,
                "batch_size": batch_size,
                **best_params,
            },
            "metrics": best_metrics,
            "dataset": dataset_name,
            "model_name": configs["model_name"],
        }
        results.append(result)
        return -best_f1

    # Create Optuna study and optimize for max_length and batch_size
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    logging.info(f"Best hyperparameters for frozen classifier on {dataset_name}: {study.best_params}")

def main():
    """Main function to run fine-tuning with hyperparameter optimization and logging."""
    global RESULTS_FILE
    args = parse_args()
    configs = load_config(args.config_file)
    setup_logging()
    set_seed(configs["seed"])

    # Set the RESULTS_FILE based on the model name
    model_name_safe = configs["model_name"].replace("/", "_")
    RESULTS_FILE = f"{model_name_safe}_best_model_results.tsv"

    # Initialize results list to store model results
    results = []

    # Process each dataset
    for dataset_config in configs["datasets"]:
        dataset_name = dataset_config["name"]
        logging.info(f"Processing dataset: {dataset_name}")
        texts_train, labels_train, texts_test, labels_test = load_sentiment_data(dataset_name)
        num_labels = len(set(labels_train))

        # Prepare datasets
        train_dataset = Dataset.from_dict({'text': texts_train, 'label': labels_train})
        if texts_test and labels_test:
            test_dataset = Dataset.from_dict({'text': texts_test, 'label': labels_test})
        else:
            test_dataset = None

        if configs.get("full_fine_tune", True):
            logging.info("Starting full fine-tuning with hyperparameter optimization and MeanPooling.")
            train_full_finetune(configs, train_dataset, test_dataset, num_labels, results, dataset_name)

        if configs.get("frozen_with_classifier", True):
            logging.info("Starting frozen model with classifier training and hyperparameter tuning.")
            # Use the provided train/test splits
            train_frozen_with_classifier(configs, texts_train, labels_train, texts_test, labels_test, results, dataset_name)

            # Save all results to the .tsv file
            save_results_to_file(results, RESULTS_FILE)

if __name__ == "__main__":
    main()