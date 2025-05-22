# LLM-SentimentProber: Toolkit for Probing Sentiment in Large Language Models
**LLM-SentimentProber** is a Python-based toolkit for analyzing and probing the hidden representations of Large Language Models (LLMs) such as LLaMA, RoBERTa, and DeBERTa for sentiment analysis. This framework empowers researchers and developers to explore how transformer models encode sentiment at each layer, train probe classifiers, and evaluate performance across benchmark datasets like SST-2, IMDB, Rotten Tomatoes, and Emotion.

## 🔍 Overview
LLM-SentimentProber enables fine-grained analysis of LLMs by extracting hidden layer representations and evaluating how well they capture sentiment-related information. The toolkit supports model-agnostic probing with various classifiers and pooling strategies and allows systematic comparisons across models and datasets.

### ✨ Key Features
* **Layer-wise Probing**: Extract and analyze hidden states from transformer layers.
* **Flexible Classifiers**: Use a range of models (e.g., logistic regression, SVM, MLP, BiLSTM, CNN) to probe hidden representations.
* **Pooling Methods**: Apply mean, max, min, last token, attention pooling, or concatenate them.
* **Multi-model & Multi-dataset**: Run experiments on LLaMA, DeBERTa, GPT, RoBERTa, and more across sentiment datasets.
* **Visualization & Evaluation**: Generate plots, confusion matrices, and statistical comparisons.
* **Extensibility**: Easily integrate new datasets, models, or probe types.

---

## Table of Contents

- [Reproducing Experimental Results](#reproducing-experimental-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models and Datasets](#supported-models-and-datasets)
- [Features](#features)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---
## Reproducing Experimental Results

Follow the steps below to set up the environment and reproduce the experimental results.

1. **Create and Activate a Conda Environment**

   Run the following commands to create a Conda environment named `ProbeToolKit` with Python 3.12 and activate it:

   ```bash
   conda create --name ProbeToolKit python=3.12 -y
   conda activate ProbeToolKit
   ```

2. **Install Dependencies**

   Install the required Python packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Hugging Face Token**

   Set your Hugging Face token as an environment variable to enable access to models and datasets. Replace `'your_huggingface_token'` with your actual token in `'hf_token.txt'` file.

   You can obtain a token from your [Hugging Face account](https://huggingface.co/settings/tokens).

4. **Run the Experiment**

   Execute the main script with the specified configuration file to reproduce the experiments:
   
   - LLama Layer-wise Exploration Experiments:
   ```bash
   python main.py --config-file reproduce_experiments.yaml
   ```
   - Fine-tuning DeBERTa 
   ```bash
   python train_encoder_classifier.py --config-file DeBERTa_config_finetuning.yaml
   ```
   - Fine-tuning RoBERTa:
   ```bash
   python train_encoder_classifier.py --config-file RoBERTa_Large_config_finetuning.yaml
   ```
   - LLama Prompting Experiments:
   ```bash
   python LLamaRunner.py
   ```

---

## 📥 Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<AnonName>/Sentiment-Probing-Toolkit-for-LLMs.git
   cd sentiment-probing-toolkit
   ```

2. **Create a Virtual Environment**:

   ```bash
   conda create --name ProbeToolKit python=3.12 -y
   conda activate ProbeToolKit
   ```

3. **Install Required Libraries**:

   Install the required dependencies listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Hugging Face API Token**:

   If you're using Hugging Face models and datasets, you need to set up a Hugging Face API token to access the models:

   Set your Hugging Face token as an environment variable to enable access to models and datasets. Replace `'your_huggingface_token'` with your actual token in `'hf_token.txt'` file.

   You can obtain a token from your [Hugging Face account](https://huggingface.co/settings/tokens).

---

## 🚀 Quick Start

Once installed, you can run the toolkit with the following command:

```bash
python your_project/main.py --config-file example_experiments.yaml
```

This will execute the experiments defined in the `example_experiments.yaml` configuration file.

---

## 🤖 Supported Models:

- **LLaMA** (e.g., LLaMA-3.2-1B-Instruct)
- **GPT** (e.g., GPT-3)
- **RoBERTa**
- **BERT**
- **Any transformer-based model from Hugging Face**

## 📊 Supported Datasets:

- **SST-2**: Stanford Sentiment Treebank
- **IMDB**: Large movie review dataset
- **Rotten Tomatoes**: Movie review dataset
- **Emotion**: Emotion dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.
- **Any sentiment dataset in CSV format with 'text' and 'label' columns**

---

## ⚙️ Features

### 🧪 Probe Classifiers
   * Logistic Regression
   * SVM (Linear/Non-linear)
   * MLP, Decision Tree, Random Forest 
   * BiLSTM, CNN 
   * LightGBM, XGBoost, Naive Bayes, KNN

### 🧠 Pooling Strategies
   * Mean, Max, Min, Last Token, Attention 
   * Concatenation of Mean + Max + Min

### 📈 Evaluation Tools
   * Accuracy, Precision, Recall, F1-score, MCC
   * Confusion Matrices 
   * Paired t-test, Wilcoxon test for statistical comparison

---

## 📁 Usage

### Running Probing Experiments

To run experiments, follow these steps:

1. **Prepare Datasets**:

   Place your dataset CSV files (e.g., `sst2_train.csv`, `sst2_test.csv`) in the `datasets/` directory. Ensure they contain `text` and `label` columns.

2. **Configure Experiments**:

   Modify the `example_experiments.yaml` file to specify models, datasets, probe types, and other options.

   ```yaml
   options:
     local_models: true # Save the model in a local folder 
     sequential: false # Execute classifier training either sequentially or concurrently
     token_level_exploration: true # Use all the pooling methods
     seed: 42
     test_on_reduced_dataset: true # Test on 100 samples 

   experiment:
     - model_name: 'meta-llama/Llama-3.2-1B-Instruct'
       dataset_name:
         - 'sst2'
         - 'rotten_tomatoes'
         - 'imdb'
       probe_types:
         - 'bilstm'
         - 'cnn'
         - 'decision-tree'
         - 'knn'
         - 'lightgbm'
         - 'linear-svm'
         - 'logistic-regression'
         - 'mlp'
         - 'naive-bayes-gaussian'
         - 'non-linear-svm'
         - 'random-forest'
         - 'xgboost'
       n_trials: 5 # Number of trials for hyperparameter optimization
       batch_size: 64
       checkpoint_path: './checkpoints'
       device: gpu
   ```

3. **Run the Toolkit**:

   Execute the `main.py` script with the configuration file:

   ```bash
   python your_project/main.py --config-file example_experiments.yaml
   ```

   This will:

   - Load the specified model(s).
   - Extract hidden states from different layers using specified pooling methods.
   - Apply the probe classifiers to the hidden states.
   - Save results, predictions, and generate evaluation reports.

---

## 🛠 Configuration

### Experiment Configuration

Experiments are configured via YAML files (e.g., `example_experiments.yaml`). Options include:

- **options**: Global settings such as `local_models`, `sequential`, `seed`, etc.
- **experiment**: List of experiments with model names, datasets, probe types, etc.

### Adding New Models or Datasets

- **Models**: Add the model name to the `model_name` field in the configuration file.
- **Datasets**: Place new dataset CSV files in the `datasets/` directory with `text` and `label` columns.

### Customizing Probe Types and Pooling Methods

- **Probe Types**: Extend the list in the `probe_types` field with custom classifiers implemented in `model_trainer.py`.
- **Pooling Methods**: Modify the `pooling_methods` list in `main.py` or configuration options.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix:

   ```bash
   git checkout -b feature/new-feature
   ```

3. **Commit your changes**:

   ```bash
   git commit -m "Add new feature"
   ```

4. **Push to your branch**:

   ```bash
   git push origin feature/new-feature
   ```

5. **Open a pull request**.

---

## 📄 License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements
- **Hugging Face Transformers** for loading pre-trained models and tokenizers.
- **Scikit-learn** for implementing various classifiers.
- **Optuna** for hyperparameter optimization.
- **PyTorch** for building neural network classifiers.
- **Matplotlib** and **Seaborn** for visualizing results.

---

## Notes

- **Logging**: All logs are saved to `experiment.log` and printed to the console.
- **Checkpoints**: Experiment progress is saved in checkpoints to allow resuming.
- **Random Seed**: Setting the `seed` ensures reproducibility.
- **Hardware Requirements**: Using large models may require significant GPU memory. Adjust `batch_size` accordingly.