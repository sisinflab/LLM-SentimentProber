import os
import pandas as pd
from datasets import load_dataset

class DownloadDataset:
    def __init__(self, dataset_name, save_dir="./datasets"):
        self.dataset_name = dataset_name
        self.save_dir = save_dir

    def load_and_process_dataset(self):
        # Load the dataset
        if self.dataset_name == 'sst2':
            dataset = load_dataset("DefenceLab/sst2")
        elif self.dataset_name == 'rotten_tomatoes':
            dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
        elif self.dataset_name == 'imdb':
            dataset = load_dataset('stanfordnlp/imdb')
        elif self.dataset_name == 'emotion':
            dataset = load_dataset('dair-ai/emotion')
        elif self.dataset_name == 'tweet_sentiment_multilingual':
            dataset = load_dataset('cardiffnlp/tweet_sentiment_multilingual', 'portuguese') # ['all', 'arabic', 'english', 'french', 'german',
                                                                                            # 'hindi', 'italian', 'portuguese', 'spanish']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Keep only 'text' and 'label' columns
        for split in dataset.keys():
            if 'text' in dataset[split].column_names:
                dataset[split] = dataset[split].select_columns(['text', 'label'])
            elif 'sentence' in dataset[split].column_names:  # For SST2
                dataset[split] = dataset[split].rename_column('sentence', 'text')
                dataset[split] = dataset[split].select_columns(['text', 'label'])
            else:
                raise ValueError(f"No 'text' or 'sentence' column found in {split} split")

        return dataset

    def save_dataset(self):
        dataset = self.load_and_process_dataset()

        # Create the directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Save each split to a CSV file
        for split in dataset.keys():
            df = pd.DataFrame(dataset[split])
            save_path = os.path.join(self.save_dir, f"{self.dataset_name}_{split}.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved {split} split to {save_path}")

# Usage example
if __name__ == "__main__":
    downloader = DownloadDataset(dataset_name="tweet_sentiment_multilingual")
    downloader.save_dataset()
