import optuna
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Tuple, List

import joblib


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is 3D by adding a seq_length dimension if it is 2D
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_size]

        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step
        return out


class CNNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is 3D by adding a seq_length dimension if it is 2D
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_size]

        x = x.permute(0, 2, 1)  # Shape: [batch_size, input_size, seq_length]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


class TrainablePoolingModel(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, seq_length: int):
        super(TrainablePoolingModel, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(seq_length, 1))
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, hidden_size]
        # Apply attention weights
        weights = torch.softmax(self.attention_weights, dim=0)  # Shape: [seq_length, 1]
        weights = weights.unsqueeze(0).transpose(1, 2)  # Shape: [1, 1, seq_length]
        x = torch.bmm(weights.repeat(x.size(0), 1, 1), x)  # Shape: [batch_size, 1, hidden_size]
        x = x.squeeze(1)  # Shape: [batch_size, hidden_size]
        out = self.fc(x)
        return out


class ModelTrainer:
    def __init__(
        self,
        model_type: str,
        hidden_states_train: np.ndarray,
        labels_train: List[int],
        hidden_states_test: np.ndarray,
        labels_test: List[int],
        n_trials: int,
        seed: int,
        on_gpu: bool,
        dataset_name: str,
        model_name: str,
        pooling_method: str,
        result_file: str = 'experiment_results.csv',
        multi_class: bool = False,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_type (str): Type of the probe model to train.
            hidden_states_train (np.ndarray): Training hidden states.
            labels_train (List[int]): Training labels.
            hidden_states_test (np.ndarray): Testing hidden states.
            labels_test (List[int]): Testing labels.
            n_trials (int): Number of trials for hyperparameter optimization.
            seed (int): Random seed.
            on_gpu (bool): Whether to use GPU.
            dataset_name (str): Name of the dataset.
            model_name (str): Name of the model.
            pooling_method (str): Pooling method used.
            result_file (str): File to save experiment results.
            multi_class (bool): Whether to perform multi-class classification.
        """
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.X_train = hidden_states_train  # Expecting numpy arrays
        self.y_train = labels_train
        self.X_test = hidden_states_test
        self.y_test = labels_test
        self.result_file = result_file
        self.n_trials = n_trials
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() and on_gpu else 'cpu')
        self.multi_class = multi_class
        logging.info(f"Using device: {self.device}")

    def save_predictions_tsv(self, y_pred: np.ndarray, layer_idx: int) -> None:
        """
        Save predictions to a TSV file for statistical analysis.

        Args:
            y_pred (np.ndarray): Predictions to save.
            layer_idx (int): Layer index.
        """
        # Define the directory path using the dataset name and model name
        dir_path = os.path.join(
            'predictions',
            self.model_name.replace('/','_'),
            self.dataset_name,
        )

        # Create the filename and full file path
        filename = f"layer_{layer_idx}_{self.pooling_method}_{self.model_type}.tsv"
        filepath = os.path.join(dir_path, filename)

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Create a DataFrame and save to TSV
        df = pd.DataFrame({'y_pred': y_pred})
        df.to_csv(filepath, sep='\t', index=False)

        logging.info(f"Saved predictions to {filepath}")

    def train_and_evaluate(self, layer_idx: int) -> Dict[str, Any]:
        """
        Train and evaluate the model.

        Args:
            layer_idx (int): Layer index.

        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        logging.info(f"Starting training for model type: {self.model_type}")

        # Handle 3D inputs for certain pooling methods
        if self.X_train.ndim == 3:
            X_train = self.X_train
            X_val = self.X_test  # We'll split further below
            y_train = self.y_train
            y_val = self.y_test
        else:
            # For 2D inputs, proceed normally
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=self.seed
            )

        # # Directory to store *all* models for each trial
        # all_models_dir = os.path.join(
        #     'weights',
        #     self.model_name.replace('/', '_'),
        #     self.dataset_name,
        #     f"layer_{layer_idx}_{self.pooling_method}_{self.model_type}_all_trials"
        # )
        # os.makedirs(all_models_dir, exist_ok=True)

        def objective(trial: optuna.Trial) -> float:
            model = self.get_model(trial)

            # Training and evaluation
            if self.model_type in ['bilstm', 'cnn', 'trainable-pooling']:
                # PyTorch model training with GPU support
                model.to(self.device)

                # Convert data to tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

                # Create DataLoaders
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(val_dataset, batch_size=256)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                )
                loss_fn = nn.CrossEntropyLoss()

                num_epochs = 5  # Adjust as needed
                for epoch in range(num_epochs):
                    model.train()
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = loss_fn(outputs, y_batch)
                        loss.backward()
                        optimizer.step()

                # Validation
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for X_batch, _ in val_loader:
                        X_batch = X_batch.to(self.device)
                        outputs = model(X_batch)
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())

                accuracy = accuracy_score(y_val, all_preds)

                # trial_model_path = os.path.join(all_models_dir, f"trial_{trial.number}.pt")
                # torch.save(model.state_dict(), trial_model_path)
                # logging.info(f"Saved trial {trial.number} model to {trial_model_path}")

                return accuracy  # Optuna maximizes this value

            else:
                # For sklearn models
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                accuracy = accuracy_score(y_val, predictions)

                # trial_model_path = os.path.join(all_models_dir, f"trial_{trial.number}.joblib")
                # joblib.dump(model, trial_model_path)
                # logging.info(f"Saved trial {trial.number} sklearn model to {trial_model_path}")

                return accuracy

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=self.n_trials)

        best_model = self.get_model(study.best_trial)

        # # Retrain on full data
        # if self.model_type in ['bilstm', 'cnn', 'trainable-pooling']:
        #     # Retrain PyTorch model on full data
        #     best_model.to(self.device)
        #
        #     X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        #     y_train_tensor = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
        #     X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        #
        #     # Create DataLoader
        #     batch_size = study.best_trial.params.get('batch_size', 64)
        #     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        #     train_loader = DataLoader(
        #         train_dataset, batch_size=batch_size, shuffle=True
        #     )
        #
        #     optimizer = torch.optim.Adam(
        #         best_model.parameters(), lr=study.best_trial.params['lr']
        #     )
        #     loss_fn = nn.CrossEntropyLoss()
        #
        #     num_epochs = 5  # Adjust as needed
        #     for epoch in range(num_epochs):
        #         best_model.train()
        #         for X_batch, y_batch in train_loader:
        #             optimizer.zero_grad()
        #             outputs = best_model(X_batch)
        #             loss = loss_fn(outputs, y_batch)
        #             loss.backward()
        #             optimizer.step()
        #
        #     # Testing
        #     best_model.eval()
        #     with torch.no_grad():
        #         outputs = best_model(X_test_tensor)
        #         preds = torch.argmax(outputs, dim=1)
        #         all_preds = preds.cpu().numpy()
        #
        #     metrics = self.evaluate_metrics(self.y_test, all_preds)
        #     # Save predictions to TSV
        #     self.save_predictions_tsv(all_preds, layer_idx)
        #
        # else:
        #     # Retrain on full data for sklearn models
        #     best_model.fit(self.X_train, self.y_train)
        #     predictions = best_model.predict(self.X_test)
        #     metrics = self.evaluate_metrics(self.y_test, predictions)
        #     # Save predictions to TSV
        #     self.save_predictions_tsv(predictions, layer_idx)

        # # Save experiment results
        # self.save_experiment_results(metrics, study.best_trial.params, layer_idx)
        # logging.info(f"Finished training for model type: {self.model_type}")
        # return metrics

        # <-- Save only the best trial -->
        # 1) Build model with best hyperparameters
        best_model = self.get_model(study.best_trial)

        # 2) Retrain best model on FULL train data
        if self.model_type in ['bilstm', 'cnn', 'trainable-pooling']:
            best_model.to(self.device)
            X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.long).to(self.device)
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

            batch_size = study.best_trial.params.get('batch_size', 64)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.Adam(
                best_model.parameters(),
                lr=study.best_trial.params['lr']
            )
            loss_fn = nn.CrossEntropyLoss()

            for _ in range(5):  # num_epochs
                best_model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = best_model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # 3) Evaluate on test set
            best_model.eval()
            with torch.no_grad():
                outputs = best_model(X_test_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # 4) Save best PyTorch model state
            best_model_dir = os.path.join(
                'models',
                self.model_name.replace('/', '_'),
                self.dataset_name
            )
            os.makedirs(best_model_dir, exist_ok=True)

            best_model_path = os.path.join(
                best_model_dir,
                f"layer_{layer_idx}_{self.pooling_method}_{self.model_type}_best.pt"
            )
            torch.save(best_model.state_dict(), best_model_path)
            logging.info(f"Best model saved to: {best_model_path}")

            metrics = self.evaluate_metrics(self.y_test, preds)
            self.save_predictions_tsv(preds, layer_idx)
        else:
            # 2) Retrain best sklearn model on full data
            best_model.fit(self.X_train, self.y_train)
            # 3) Evaluate on test set
            predictions = best_model.predict(self.X_test)

            # 4) Save best sklearn model
            best_model_dir = os.path.join(
                'weights',
                self.model_name.replace('/', '_'),
                self.dataset_name
            )
            os.makedirs(best_model_dir, exist_ok=True)

            best_model_path = os.path.join(
                best_model_dir,
                f"layer_{layer_idx}_{self.pooling_method}_{self.model_type}_best.joblib"
            )
            joblib.dump(best_model, best_model_path)
            logging.info(f"Best sklearn model saved to: {best_model_path}")

            metrics = self.evaluate_metrics(self.y_test, predictions)
            self.save_predictions_tsv(predictions, layer_idx)

        # Save final results
        self.save_experiment_results(metrics, study.best_trial.params, layer_idx)
        logging.info(f"Finished training for model type: {self.model_type}")
        return metrics

    def get_model(self, trial: optuna.Trial):
        """
        Get the model based on the trial parameters.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            The initialized model.
        """
        input_dim = self.X_train.shape[-1]  # Adjusted to get input dimension
        output_dim = len(np.unique(self.y_train))  # Dynamically determine number of classes

        if self.model_type == 'logistic-regression':
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=self.seed,
                multi_class='multinomial' if self.multi_class else 'auto',
                solver='lbfgs',
            )

        elif self.model_type == 'linear-svm':
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LinearSVC(
                C=C,
                max_iter=1000,
                random_state=self.seed,
                multi_class='ovr',
            )

        elif self.model_type == 'mlp':
            hidden_layer_sizes = trial.suggest_categorical(
                'hidden_layer_sizes', [(64,), (128,), (64, 64), (128, 64)]
            )
            alpha = trial.suggest_float('alpha', 1e-4, 1e-1, log=True)
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=1000,
                random_state=self.seed,
            )

        elif self.model_type == 'non-linear-svm':
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
            model = SVC(
                C=C,
                kernel=kernel,
                probability=True,
                random_state=self.seed,
            )

        elif self.model_type == 'decision-tree':
            max_depth = trial.suggest_int('max_depth', 3, 30)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=self.seed)

        elif self.model_type == 'random-forest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=self.seed
            )

        elif self.model_type == 'xgboost':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1, log=True)
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='mlogloss',
                objective='multi:softprob' if self.multi_class else 'binary:logistic',
                num_class=output_dim if self.multi_class else None,
            )

        elif self.model_type == 'lightgbm':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1, log=True)
            model = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.seed,
                objective='multiclass' if self.multi_class else 'binary',
                num_class=output_dim if self.multi_class else None,
            )

        elif self.model_type == 'naive-bayes-gaussian':
            model = GaussianNB()

        elif self.model_type == 'knn':
            n_neighbors = trial.suggest_int('n_neighbors', 2, output_dim)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif self.model_type == 'bilstm':
            hidden_size = trial.suggest_int('hidden_size', 50, 200)
            model = BiLSTM(
                input_size=input_dim, hidden_size=hidden_size, output_size=output_dim
            )

        elif self.model_type == 'cnn':
            model = CNNClassifier(
                input_size=input_dim, num_classes=output_dim
            )

        elif self.model_type == 'trainable-pooling':
            seq_length = self.X_train.shape[1]  # Sequence length
            hidden_size = self.X_train.shape[2]  # Hidden size
            model = TrainablePoolingModel(
                hidden_size=hidden_size,
                num_classes=output_dim,
                seq_length=seq_length,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def evaluate_metrics(self, y_true: List[int], y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate various metrics.

        Args:
            y_true (List[int]): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, Any]: Calculated metrics.
        """
        average_method = 'macro' if self.multi_class else 'binary'

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average=average_method, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average=average_method, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average=average_method, zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': conf_matrix,
        }

    def save_experiment_results(
        self, metrics: Dict[str, Any], params: Dict[str, Any], layer_idx: int
    ) -> None:
        """
        Save the results of the best model to a CSV file.

        Args:
            metrics (Dict[str, Any]): Evaluation metrics.
            params (Dict[str, Any]): Hyperparameters.
            layer_idx (int): Layer index.
        """
        # Convert the confusion matrix to a string representation
        conf_matrix_str = ','.join(map(str, metrics['confusion_matrix'].flatten()))

        # Create a dictionary of metrics and parameters
        results = {
            'layer': layer_idx,
            'model_type': self.model_type,
            'pooling_method': self.pooling_method,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'mcc': metrics['mcc'],
            'confusion_matrix': conf_matrix_str,
        }

        # Add the hyperparameters to the results
        results.update(params)

        # Ensure the directory exists before saving the file
        result_dir = os.path.dirname(self.result_file)
        os.makedirs(result_dir, exist_ok=True)

        # Check if the result file exists
        if not os.path.exists(self.result_file):
            # Create a new DataFrame and save it as a CSV file
            df = pd.DataFrame([results])
            df.to_csv(self.result_file, index=False)
        else:
            # Load the existing CSV file and append the new results
            df = pd.read_csv(self.result_file)
            # Concatenate the new results row to the existing DataFrame
            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
            df.to_csv(self.result_file, index=False)

        # Log the success message
        logging.info(f"Saved experiment results to {self.result_file}")