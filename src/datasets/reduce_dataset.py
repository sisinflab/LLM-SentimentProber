import pandas as pd
import numpy as np

def reduce_dataset(input_csv, output_csv, total_samples, max_avg_length, random_seed=42):
    """
    Reduces the dataset to a specified number of samples while maintaining statistical consistency.

    Parameters:
    - input_csv (str): Path to the input CSV file with at least 'text' and 'label' columns.
    - output_csv (str): Path to save the reduced dataset CSV.
    - total_samples (int): Total number of samples to retain.
    - max_avg_length (float): Maximum average sentence length (in words) across selected samples.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - None
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Load the dataset
    df = pd.read_csv(input_csv)

    # Check if dataset is large enough
    if total_samples > len(df):
        raise ValueError("Total samples requested exceed the size of the dataset.")

    # Calculate initial statistics
    original_label_dist = df['label'].value_counts(normalize=True)
    df['sentence_length'] = df['text'].apply(lambda x: len(str(x).split()))
    original_avg_length = df['sentence_length'].mean()

    # Initialize variables
    reduced_df = pd.DataFrame()
    labels = df['label'].unique()

    # Calculate samples per label to maintain label distribution
    samples_per_label = {}
    for label in labels:
        label_fraction = original_label_dist[label]
        samples_per_label[label] = int(total_samples * label_fraction)

    # Handle edge cases where rounding might cause total samples to differ
    total_allocated = sum(samples_per_label.values())
    difference = total_samples - total_allocated

    # Adjust sample counts to match total_samples
    while difference != 0:
        for label in labels:
            if difference == 0:
                break
            samples_per_label[label] += 1 if difference > 0 else -1
            difference += -1 if difference > 0 else 1

    # Sample data for each label
    for label in labels:
        label_df = df[df['label'] == label].copy()

        # Filter samples to meet the max average sentence length constraint
        label_avg_length = label_df['sentence_length'].mean()
        if label_avg_length > max_avg_length:
            # Sort by sentence length
            label_df = label_df.sort_values(by='sentence_length')
            # Select samples with shorter sentences
            cumulative_avg_length = 0
            selected_indices = []
            for idx, row in label_df.iterrows():
                selected_indices.append(idx)
                cumulative_avg_length += row['sentence_length']
                current_avg_length = cumulative_avg_length / len(selected_indices)
                if current_avg_length > max_avg_length:
                    selected_indices.pop()
                    break
                if len(selected_indices) == samples_per_label[label]:
                    break
            label_sample = label_df.loc[selected_indices]
        else:
            # Randomly sample if average length is within constraints
            label_sample = label_df.sample(n=samples_per_label[label], random_state=random_seed)

        reduced_df = pd.concat([reduced_df, label_sample], ignore_index=True)

    # Shuffle the reduced dataset
    reduced_df = reduced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Recalculate statistics for the reduced dataset
    reduced_label_dist = reduced_df['label'].value_counts(normalize=True)
    reduced_avg_length = reduced_df['sentence_length'].mean()

    # Check if constraints are met
    if reduced_avg_length > max_avg_length:
        raise ValueError("Could not meet the maximum average sentence length constraint without distorting the dataset significantly.")

    # Save the reduced dataset
    reduced_df.drop(columns=['sentence_length'], inplace=True)
    reduced_df.to_csv(output_csv, index=False)

    # Print summary
    print("Original Dataset:")
    print(f"- Total samples: {len(df)}")
    print(f"- Label distribution:\n{original_label_dist}")
    print(f"- Average sentence length: {original_avg_length:.2f} words\n")

    print("Reduced Dataset:")
    print(f"- Total samples: {len(reduced_df)}")
    print(f"- Label distribution:\n{reduced_label_dist}")
    print(f"- Average sentence length: {reduced_avg_length:.2f} words")

# Example usage:
if __name__ == '__main__':
    reduce_dataset('imdb_test.csv', 'reduced_imdb_test.csv', total_samples=6250, max_avg_length=100)