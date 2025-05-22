import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from utils import load_sentiment_data
from tqdm import tqdm  # Added for progress bars

def evaluate_predictions(model_name: str, predictions_dir: str, y_true: np.ndarray, multi_class: bool = False) -> None:
    """
    Evaluate predictions from different layers and pooling methods,
    compute metrics, generate plots, and perform statistical tests.

    Args:
        model_name (str): Name of the model.
        predictions_dir (str): Directory containing prediction TSV files.
        y_true (np.ndarray): Ground truth labels.
        multi_class (bool): Whether to perform multi-class evaluation.
    """
    print(f"Evaluating predictions in directory: {predictions_dir}")

    # Collect all prediction files
    predictions_files = [f for f in os.listdir(predictions_dir) if f.endswith('.tsv')]

    if not predictions_files:
        logging.error("No prediction files found for evaluation.")
        print("No prediction files found for evaluation.")
        return

    # Prepare a list to collect all results
    all_results = []

    # Loop with tqdm for progress bar
    for file in tqdm(predictions_files, desc="Processing prediction files"):
        # Extract model information from the file name
        # Expected filename format: layer_{layer_idx}_{pooling_method}_{probe_type}.tsv
        filename_parts = file.replace('.tsv', '').split('_')
        if len(filename_parts) < 4:
            logging.error(f"Filename {file} does not match expected format. Skipping.")
            print(f"Filename {file} does not match expected format. Skipping.")
            continue

        layer_idx = filename_parts[1]
        pooling_method = filename_parts[2]
        probe_type = '_'.join(filename_parts[3:])  # In case probe_type has underscores
        model_type = model_name

        filepath = os.path.join(predictions_dir, file)
        try:
            df = pd.read_csv(filepath, sep='\t')
            if 'y_pred' not in df.columns:
                logging.error(f"'y_pred' column not found in {file}. Skipping.")
                print(f"'y_pred' column not found in {file}. Skipping.")
                continue
            preds = df['y_pred'].values
            # Align lengths
            min_len = min(len(preds), len(y_true))
            preds = preds[:min_len]
            y_true_aligned = y_true[:min_len]
            # Compute metrics
            average_method = 'macro' if multi_class else 'binary'
            accuracy = accuracy_score(y_true_aligned, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_aligned, preds, average=average_method, zero_division=0
            )
            mcc = matthews_corrcoef(y_true_aligned, preds)
            # Store the results
            all_results.append({
                'model_type': model_type,
                'layer_idx': int(layer_idx),
                'pooling_method': pooling_method,
                'probe_type': probe_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'preds': preds,
                'y_true': y_true_aligned,
            })
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            print(f"Error reading {file}: {e}")
            continue

    if not all_results:
        logging.error("No valid predictions data found for evaluation.")
        print("No valid predictions data found for evaluation.")
        return

    # Create evaluation directory
    evaluation_dir = os.path.join(predictions_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

    # Convert all_results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # Define consistent color palettes
    probe_types = sorted(results_df['probe_type'].unique())
    pooling_methods = sorted(results_df['pooling_method'].unique())

    # Create a color palette for probe_types
    # Using a custom color palette for better distinguishability
    probe_colorpalette = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261', '#264653', '#e9c46a',
                            '#9d5a6c', '#a8dadc', '#5ffad3', '#ffcbf2', '#6a0572', '#ff6f61']
    probe_palette = dict(zip(
        probe_types,
        sns.color_palette(probe_colorpalette, n_colors=len(probe_types))
    ))

    # Create a color palette for pooling_methods
    # Using a custom color palette for visual appeal
    pooling_colorpalette = ['#ef476f', '#f78c6b', '#ffd166', '#06d6a0', '#118ab2', '#073b4c']
    pooling_palette = dict(zip(
        pooling_methods,
        sns.color_palette(pooling_colorpalette, n_colors=len(pooling_methods))
    ))

    # Save evaluation results to CSV
    results_csv_path = os.path.join(evaluation_dir, 'evaluation_results.csv')
    results_df.drop(columns=['preds', 'y_true']).to_csv(results_csv_path, index=False)
    logging.info(f"\nSaved evaluation results to '{results_csv_path}'.")
    print(f"\nSaved evaluation results to '{results_csv_path}'.")

    # Generate plots
    print("Generating plots...")
    # plot_confusion_matrices(results_df, evaluation_dir)
    # plot_overall_metrics(results_df, evaluation_dir)
    plot_per_layer_accuracy_by_pooling(results_df, evaluation_dir, probe_palette)
    # plot_per_pooling_method_metrics_by_layer(results_df, evaluation_dir, probe_palette)
    # plot_layer_vs_pooling_method_interaction(results_df, evaluation_dir, pooling_palette)
    # plot_per_probe_accuracy_by_pooling(results_df, evaluation_dir, pooling_palette)
    print("\nPlot generation complete.\n\n")

    # Perform statistical tests
    print("Performing statistical tests...")
    perform_statistical_tests(results_df, evaluation_dir)
    print("Statistical tests complete.")

def evaluate_predictions_prompt(model_name: str, predictions_dir: str, y_true: np.ndarray, multi_class: bool = False) -> None:
    """
    Evaluate predictions from prompt-based methods,
    compute metrics, and save results.

    Args:
        model_name (str): Name of the model.
        predictions_dir (str): Directory containing prediction TSV files.
        y_true (np.ndarray): Ground truth labels.
        multi_class (bool): Whether to perform multi-class evaluation.
    """
    print(f"Evaluating prompt-based predictions in directory: {predictions_dir}")

    # Collect all prediction files
    predictions_files = [f for f in os.listdir(predictions_dir) if f.endswith('.tsv')]

    if not predictions_files:
        logging.error("No prediction files found for evaluation.")
        print("No prediction files found for evaluation.")
        return

    # Prepare a list to collect all results
    all_results = []

    # Loop with tqdm for progress bar
    for file in tqdm(predictions_files, desc="Processing prompt prediction files"):
        # Extract prompt type from the file name
        # Expected filename format: {prompt_type}.tsv
        filename_parts = file.replace('.tsv', '').split('_')
        prompt_type = '_'.join(filename_parts)  # In case prompt_type has underscores
        model_type = model_name

        filepath = os.path.join(predictions_dir, file)
        try:
            df = pd.read_csv(filepath, sep='\t')
            if 'y_pred' not in df.columns:
                logging.error(f"'y_pred' column not found in {file}. Skipping.")
                print(f"'y_pred' column not found in {file}. Skipping.")
                continue
            preds = df['y_pred'].values
            # Align lengths
            min_len = min(len(preds), len(y_true))
            preds = preds[:min_len]
            y_true_aligned = y_true[:min_len]
            # Compute metrics
            average_method = 'macro' if multi_class else 'binary'
            accuracy = accuracy_score(y_true_aligned, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_aligned, preds, average=average_method, zero_division=0
            )
            mcc = matthews_corrcoef(y_true_aligned, preds)
            # Store the results
            all_results.append({
                'model_type': model_type,
                'prompt_type': prompt_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'preds': preds,
                'y_true': y_true_aligned,
            })
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            print(f"Error reading {file}: {e}")
            continue

    if not all_results:
        logging.error("No valid predictions data found for evaluation.")
        print("No valid predictions data found for evaluation.")
        return

    # Create evaluation directory
    evaluation_dir = os.path.join(predictions_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

    # Convert all_results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # Define consistent color palette for prompt types
    prompt_types = results_df['prompt_type'].unique()
    prompt_palette = dict(zip(
        prompt_types,
        sns.color_palette('colorblind', n_colors=len(prompt_types))
    ))

    # Save evaluation results to CSV
    results_csv_path = os.path.join(evaluation_dir, 'evaluation_results.csv')
    results_df.drop(columns=['preds', 'y_true']).to_csv(results_csv_path, index=False)
    logging.info(f"\nSaved evaluation results to '{results_csv_path}'.")
    print(f"\nSaved evaluation results to '{results_csv_path}'.")

    # You can add plotting and statistical testing for prompt-based evaluations as needed
    print("Prompt-based evaluation complete.")

def plot_confusion_matrices(results_df: pd.DataFrame, evaluation_dir: str) -> None:
    """
    Plot confusion matrices for each probe type, pooling method, and each layer separately.
    Saves plots in specific directories for each probe type and pooling method under 'confusion_matrices/'.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
    """
    print("Plotting confusion matrices...")
    # Ensure the root directory exists
    base_dir = os.path.join(evaluation_dir, "confusion_matrices")
    os.makedirs(base_dir, exist_ok=True)

    # Iterate over each unique probe type
    for probe in tqdm(results_df['probe_type'].unique(), desc="Probes"):
        probe_results = results_df[results_df['probe_type'] == probe]

        # Iterate over each unique pooling method
        for pooling_method in tqdm(probe_results['pooling_method'].unique(), desc=f"Pooling Methods for Probe {probe}", leave=False):
            pooling_results = probe_results[probe_results['pooling_method'] == pooling_method]

            # Create a specific directory for this probe type and pooling method
            probe_dir = os.path.join(base_dir, f'confusion_matrix_{probe}_{pooling_method}')
            os.makedirs(probe_dir, exist_ok=True)

            # Iterate over each unique layer
            for layer in pooling_results['layer_idx'].unique():
                layer_results = pooling_results[pooling_results['layer_idx'] == layer]

                if layer_results.empty:
                    continue  # Skip if no results for this combination

                # Since preds and y_true are arrays, we need to concatenate them
                preds = np.concatenate(layer_results['preds'].values)
                y_true = np.concatenate(layer_results['y_true'].values)

                # Generate the confusion matrix
                cm = confusion_matrix(y_true, preds)

                # Plot confusion matrix with a more polished design
                fig, ax = plt.subplots(figsize=(6, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues', ax=ax, colorbar=False)

                # Customize the plot for a cleaner appearance
                ax.set_title(f'Confusion Matrix\nProbe: {probe} | Pooling: {pooling_method} | Layer: {layer}', fontsize=14, pad=20)
                ax.set_xlabel('Predicted Labels', fontsize=12)
                ax.set_ylabel('True Labels', fontsize=12)
                plt.grid(False)  # Disable grid to keep the focus on the matrix
                plt.tight_layout()

                # Save the plot in the specific directory
                plot_filename = os.path.join(probe_dir, f'confusion_matrix_{probe}_pooling_{pooling_method}_layer_{layer}.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close()

                # Logging the saved file
                logging.info(f"Saved confusion matrix for probe '{probe}', pooling '{pooling_method}', layer '{layer}' as '{plot_filename}'.")
    print("\nConfusion matrices plotting complete.")

def plot_overall_metrics(results_df: pd.DataFrame, evaluation_dir: str) -> None:
    """
    Plot overall metrics comparison grouped by both layer and pooling method, comparing all probe types.
    Adds a dotted horizontal line for the best model by accuracy, and places the legend outside the plot.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
    """
    print("Plotting overall metrics...")
    # Ensure a base directory exists for overall metrics
    base_dir = os.path.join(evaluation_dir, "overall_metrics")
    os.makedirs(base_dir, exist_ok=True)

    # Define the numeric columns for the metrics
    numeric_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']

    # Define a consistent color palette for metrics
    metric_palette = dict(zip(
        numeric_columns,
        sns.color_palette('colorblind', n_colors=len(numeric_columns))
    ))

    # Get unique layers, pooling methods, and probe types
    unique_layers = sorted(results_df['layer_idx'].unique())
    unique_pooling_methods = sorted(results_df['pooling_method'].unique())

    # Iterate over each layer
    for layer in tqdm(unique_layers, desc="Layers"):
        for pooling_method in tqdm(unique_pooling_methods, desc=f"Pooling Methods for Layer {layer}", leave=False):
            # Filter the results for the current layer and pooling method
            layer_pooling_results = results_df[
                (results_df['layer_idx'] == layer) &
                (results_df['pooling_method'] == pooling_method)
            ]

            if layer_pooling_results.empty:
                continue  # Skip if no results for this layer and pooling method combination

            # Melt the DataFrame for seaborn plotting
            metrics_melted = layer_pooling_results.melt(
                id_vars=['probe_type', 'layer_idx', 'pooling_method'],
                value_vars=numeric_columns,
                var_name='Metric',
                value_name='Value',
            )

            # Find the best accuracy value for the current layer and pooling method
            best_accuracy = layer_pooling_results['accuracy'].max()

            # Set up the figure with a larger size for better visibility
            plt.figure(figsize=(12, 8))

            # Plot the metrics comparison for each layer and pooling method
            sns.barplot(
                x='probe_type',
                y='Value',
                hue='Metric',
                data=metrics_melted,
                ci=None,
                palette=metric_palette
            )

            # Add a horizontal dotted line at the best accuracy
            plt.axhline(y=best_accuracy, color='red', linestyle='--', label=f'Best accuracy: {best_accuracy:.2f}')

            # Customize labels, titles, and ticks for better readability
            plt.title(f'Metrics Comparison (Layer {layer}, Pooling: {pooling_method})', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Probe Type', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)

            # Move the legend outside the plot, with smaller font for clarity
            plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
            plt.tight_layout()

            # Save plot with the pooling method in the filename
            plot_filename = os.path.join(base_dir, f'overall_metrics_layer_{layer}_pooling_{pooling_method}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            # Logging
            logging.info(f"Saved metrics comparison plot for layer '{layer}', pooling '{pooling_method}' as '{plot_filename}'.")
    print("\nOverall metrics plotting complete.")

def plot_per_layer_accuracy_by_pooling(results_df: pd.DataFrame, evaluation_dir: str, probe_palette: dict) -> None:
    """
    Plot per-layer accuracy comparison for each pooling method, highlighting the best accuracy with a star marker.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
        probe_palette (dict): Color palette for probe types.
    """
    print("Plotting per-layer accuracy by pooling method...")
    # Ensure the directory exists
    base_dir = os.path.join(evaluation_dir, "per_layer_accuracy_by_pooling")
    os.makedirs(base_dir, exist_ok=True)

    # Define the metric to plot (accuracy)
    metric = 'accuracy'

    # Get unique pooling methods
    unique_pooling_methods = sorted(results_df['pooling_method'].unique())

    # Iterate over each pooling method to create separate plots
    for pooling_method in tqdm(unique_pooling_methods, desc="Pooling Methods"):
        # Filter the DataFrame by pooling method
        pooling_df = results_df[results_df['pooling_method'] == pooling_method]

        if pooling_df.empty:
            continue  # Skip if no data for this pooling method

        # Calculate the best accuracy and corresponding layer/probe type
        best_accuracy_idx = pooling_df[metric].idxmax()
        best_accuracy = pooling_df.loc[best_accuracy_idx, metric]
        best_layer = pooling_df.loc[best_accuracy_idx, 'layer_idx']
        best_probe = pooling_df.loc[best_accuracy_idx, 'probe_type']

        # Set up the figure with larger size and tight layout
        plt.figure(figsize=(8, 8))

        # Get the list of probe types to ensure consistent legend order
        probe_types = sorted(pooling_df['probe_type'].unique().tolist())

        # Plot the accuracy comparison for all probe types
        sns.lineplot(
            x='layer_idx',
            y=metric,
            hue='probe_type',
            hue_order=probe_types,  # Ensures the legend order matches probe_types
            data=pooling_df,
            marker='o',
            linewidth=2.5,  # Thicker lines for better visibility
            palette=probe_palette,
            legend=False
        )

        # Highlight the best accuracy with a larger star marker using the same color
        plt.scatter(
            x=[best_layer],
            y=[best_accuracy],
            color='red',
            marker='*',
            s=200,  # Increase size of the star marker for better visibility
            zorder=5,  # Ensure the marker appears on top
            edgecolors='black',
            linewidths=1
            #label=f'Best accuracy: {best_accuracy:.2f}\n(Layer {best_layer}, Probe: {best_probe})'
        )

        # Add a label next to the star marker
        plt.text(
            x=best_layer + 0.5,  # Slightly offset to the right of the marker
            y=best_accuracy,
            s=f"Best accuracy: {best_accuracy:.2f}\n(Layer {best_layer}, Probe: {best_probe})",
            fontsize=10,
            color='black',
            ha='left',  # Align text to the left
            va='center',  # Vertically align text to the center
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')  # Add a background box
        )

        # Add grid for better visibility of values
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Customize the title, axis labels, and ticks
        # plt.title(f'Per-Layer Probe {metric.capitalize()} Comparison (Pooling: {pooling_method})', fontsize=16,
        #           fontweight='bold', pad=20)
        plt.xlabel('Layer Index', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Adjust the legend to avoid duplication
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), title='Probe Type',
        #            bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=10)

        # Ensure tight layout to avoid overlapping or cutoff elements
        plt.tight_layout()

        # Save plot with the pooling method in the filename
        plot_filename = os.path.join(base_dir, f'per_layer_probe_{metric}_comparison_{pooling_method}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        # plt.close()

        # Save as PDF for LaTeX
        plot_filename = os.path.join(base_dir, f'per_layer_probe_{metric}_comparison_{pooling_method}.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()

        # Logging
        logging.info(f"Saved per-layer {metric} comparison plot for pooling '{pooling_method}' as '{plot_filename}'.")
    print("\nPer-layer accuracy plotting complete.")

def plot_per_pooling_method_metrics_by_layer(results_df: pd.DataFrame, evaluation_dir: str, probe_palette: dict) -> None:
    """
    Plot per-pooling-method metrics comparison for each layer, with a horizontal dotted line for best accuracy.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
        probe_palette (dict): Color palette for probe types.
    """
    print("Plotting per-pooling-method metrics by layer...")
    base_dir = os.path.join(evaluation_dir, "per_pooling_accuracy_by_pooling")
    os.makedirs(base_dir, exist_ok=True)

    # Define the numeric columns (metrics)
    numeric_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']

    # Get unique layers
    unique_layers = sorted(results_df['layer_idx'].unique())

    # Iterate over each layer
    for layer in tqdm(unique_layers, desc="Layers for per-pooling-method metrics"):
        # Filter data by the current layer
        layer_df = results_df[results_df['layer_idx'] == layer]

        if layer_df.empty:
            continue  # Skip if no data for this layer

        # Aggregate metrics over layers for each pooling method and probe_type within the current layer
        pooling_metrics_df = layer_df.groupby(['pooling_method', 'probe_type'])[numeric_columns].mean().reset_index()

        # Iterate over each metric and plot separately
        for metric in numeric_columns:
            plt.figure(figsize=(12, 8))

            # Create the barplot for each metric
            sns.barplot(x='pooling_method', y=metric, hue='probe_type', data=pooling_metrics_df, palette=probe_palette)

            # Add title and labels with improved formatting
            plt.title(f'{metric.capitalize()} Comparison Across Pooling Methods (Layer {layer})', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Pooling Method', fontsize=12)
            plt.ylabel(metric.capitalize(), fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Add a grid to improve readability
            plt.grid(True, linestyle='--', alpha=0.6)

            # If metric is accuracy, add the horizontal dotted line for best accuracy
            if metric == 'accuracy':
                best_accuracy = pooling_metrics_df[metric].max()
                best_accuracy_idx = pooling_metrics_df[metric].idxmax()
                best_pooling = pooling_metrics_df.loc[best_accuracy_idx, 'pooling_method']

                plt.axhline(y=best_accuracy, color='red', linestyle='--', label=f'Best accuracy: {best_accuracy:.2f} (Pooling: {best_pooling})')

            # Adjust the legend - move outside for better visibility
            plt.legend(title='Probe Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            # Save the plot with tight layout and a high-resolution dpi
            plt.tight_layout()
            plot_filename = os.path.join(base_dir, f'per_pooling_method_{metric}_comparison_layer_{layer}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            # Logging the saved plot
            logging.info(f"Saved per-pooling-method {metric} comparison plot for layer {layer} as '{plot_filename}'.")
    print("\nPer-pooling-method metrics plotting complete.")

def plot_layer_vs_pooling_method_interaction(results_df: pd.DataFrame, evaluation_dir: str, pooling_palette: dict) -> None:
    """
    Plot an interaction plot to visualize the interaction between layers and pooling methods for accuracy.
    Highlights the best accuracy with a red star marker.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
        pooling_palette (dict): Color palette for pooling methods.
    """
    print("Plotting layer vs pooling method interaction...")
    base_dir = os.path.join(evaluation_dir, "layer_vs_pooling_method_interaction")
    os.makedirs(base_dir, exist_ok=True)

    # Set figure size and Seaborn style for better aesthetics
    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")

    # Ensure consistent order of pooling methods
    pooling_methods = sorted(pooling_palette.keys())

    # Create the line plot for layer vs pooling method interaction
    sns.lineplot(
        x='layer_idx',
        y='accuracy',
        hue='pooling_method',
        hue_order=pooling_methods,  # Ensures consistent hue order
        data=results_df,
        marker='o',       # Add markers to highlight each point
        linewidth=2.5,    # Thicker lines for better visibility
        palette=pooling_palette,
        legend=False  # Disable legend
    )

    # Add a title and labels with improved formatting
    # plt.title("Layer vs Pooling Method Interaction Plot", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid to enhance readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust the legend to maintain correct label-handle mapping
    # handles, labels = plt.gca().get_legend_handles_labels()
    # legend_dict = dict(zip(labels, handles))
    # sorted_labels = sorted(legend_dict.keys())
    # sorted_handles = [legend_dict[label] for label in sorted_labels]
    # plt.legend(
    #     sorted_handles,
    #     sorted_labels,
    #     title='Pooling Method',
    #     bbox_to_anchor=(1.05, 1),
    #     loc='upper left',
    #     fontsize=10
    # )

    # Tighten layout and save plot with high DPI for better quality
    plt.tight_layout()
    plot_filename = os.path.join(base_dir, 'layer_vs_pooling_method_interaction.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # plt.close()

    # Save as PDF for LaTeX
    plot_filename = os.path.join(base_dir, 'layer_vs_pooling_method_interaction.pdf')
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Plot saved as PDF: {plot_filename}")

    # Logging for successful plot saving
    logging.info(f"Saved layer vs pooling method interaction plot as '{plot_filename}'.")
    print("\nLayer vs pooling method interaction plotting complete.")


def plot_per_probe_accuracy_by_pooling(results_df: pd.DataFrame, evaluation_dir: str, pooling_palette: dict) -> None:
    """
    Plot per-probe accuracy comparison across pooling methods, highlighting the best accuracy with a star marker.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save plots.
        pooling_palette (dict): Color palette for pooling methods.
    """
    print("Plotting per-probe accuracy by pooling method...")
    base_dir = os.path.join(evaluation_dir, "per_probe_accuracy_by_pooling")
    os.makedirs(base_dir, exist_ok=True)

    # Define the metric to plot (accuracy)
    metric = 'accuracy'

    # Get unique probe types and pooling methods
    unique_probes = sorted(results_df['probe_type'].unique())
    unique_pooling_methods = sorted(results_df['pooling_method'].unique())

    # Iterate over each probe type to create separate plots
    for probe in tqdm(unique_probes, desc="Probes for per-probe accuracy"):
        probe_df = results_df[results_df['probe_type'] == probe]

        if probe_df.empty:
            continue  # Skip if no data for this probe type

        plt.figure(figsize=(14, 8))  # Increase figure size for clarity

        # Iterate over each pooling method and create lines in the plot
        for pooling_method in unique_pooling_methods:
            pooling_df = probe_df[probe_df['pooling_method'] == pooling_method]

            if pooling_df.empty:
                continue  # Skip if no data for this pooling method

            # Plot the per-layer accuracy comparison for the current probe and pooling method
            sns.lineplot(
                x='layer_idx',
                y=metric,
                data=pooling_df,
                label=pooling_method,
                marker='o',
                markersize=8,  # Increase marker size for better visibility
                linewidth=2.5,  # Increase line width for better visibility
                color=pooling_palette[pooling_method]
            )

            # Calculate the best accuracy and corresponding layer for the current pooling method
            best_accuracy_idx = pooling_df[metric].idxmax()
            best_accuracy = pooling_df.loc[best_accuracy_idx, metric]
            best_layer = pooling_df.loc[best_accuracy_idx, 'layer_idx']

            # Highlight the best accuracy with a larger star marker
            plt.scatter(
                x=[best_layer],
                y=[best_accuracy],
                color=pooling_palette[pooling_method],
                marker='*',
                s=250,  # Larger star size for emphasis
                zorder=5,  # Ensure the marker is on top
                edgecolors='black',
                linewidths=1,
                label=f'Best accuracy ({pooling_method}): {best_accuracy:.2f}\n(Layer {best_layer})'
            )

        # Add grid for better visibility
        plt.grid(True, which='both', linestyle='--', linewidth=0.6)

        # Title and layout adjustments
        plt.title(f'Per-Probe Accuracy Comparison Across Pooling Methods (Probe: {probe})', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Layer Index', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)

        # Customize ticks and labels for improved readability
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Adjust the legend to avoid duplication
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Pooling Method',
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save plot with the probe type in the filename
        plot_filename = os.path.join(base_dir, f'per_probe_accuracy_comparison_{probe}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')  # Save with high resolution and tight layout
        plt.close()

        # Logging
        logging.info(f"Saved per-probe accuracy comparison plot for probe '{probe}' as '{plot_filename}'.")
    print("\nPer-probe accuracy plotting complete.")

def perform_statistical_tests(results_df: pd.DataFrame, evaluation_dir: str) -> None:
    """
    Perform statistical tests between probes and save results.

    Args:
        results_df (DataFrame): DataFrame containing results.
        evaluation_dir (str): Directory to save results.
    """
    print("Performing statistical tests...")
    probes = results_df['probe_type'].unique()

    # Prepare lists to collect test results
    t_test_results = []
    wilcoxon_results = []

    # For each layer and pooling method, perform pairwise statistical tests
    layers = sorted(results_df['layer_idx'].unique())
    pooling_methods = sorted(results_df['pooling_method'].unique())

    try:
        for layer in tqdm(layers, desc="Layers for statistical tests"):
            for pooling_method in pooling_methods:
                subset = results_df[
                    (results_df['layer_idx'] == layer) & (results_df['pooling_method'] == pooling_method)
                ]
                if subset.empty:
                    continue

                # Prepare predictions for statistical tests
                predictions_dict = {row['probe_type']: row['preds'] for _, row in subset.iterrows()}
                y_true = subset.iloc[0]['y_true']

                probe_list = list(predictions_dict.keys())
                for i in range(len(probe_list)):
                    for j in range(i + 1, len(probe_list)):
                        probe1 = probe_list[i]
                        probe2 = probe_list[j]
                        preds1 = predictions_dict[probe1]
                        preds2 = predictions_dict[probe2]

                        min_len = min(len(preds1), len(preds2))
                        preds1 = preds1[:min_len]
                        preds2 = preds2[:min_len]
                        y_true_aligned = y_true[:min_len]

                        # Convert predictions to correctness (1 for correct, 0 for incorrect)
                        acc1 = (preds1 == y_true_aligned).astype(int)
                        acc2 = (preds2 == y_true_aligned).astype(int)

                        differences = acc1 - acc2

                        # Paired t-test
                        try:
                            if np.all(differences == 0):
                                t_stat = 0.0
                                t_p_value = 1.0
                                logging.info(
                                    f"Layer {layer}, Pooling {pooling_method}: Paired t-test between {probe1} and {probe2}: All differences are zero."
                                )
                            else:
                                t_stat, t_p_value = ttest_rel(acc1, acc2)
                                logging.info(
                                    f"Layer {layer}, Pooling {pooling_method}: Paired t-test between {probe1} and {probe2}: t-stat={t_stat:.4f}, p-value={t_p_value:.4f}"
                                )
                            t_test_results.append({
                                'Layer': layer,
                                'Pooling Method': pooling_method,
                                'Probe 1': probe1,
                                'Probe 2': probe2,
                                't-stat': t_stat,
                                'p-value': t_p_value,
                            })
                        except Exception as e:
                            logging.error(
                                f"Error performing t-test between {probe1} and {probe2} at layer {layer}, pooling {pooling_method}: {e}"
                            )

                        # Wilcoxon signed-rank test
                        try:
                            if np.all(differences == 0):
                                wilcoxon_stat = 0.0
                                wilcoxon_p_value = 1.0
                                logging.info(
                                    f"Layer {layer}, Pooling {pooling_method}: Wilcoxon test between {probe1} and {probe2}: All differences are zero."
                                )
                            else:
                                wilcoxon_stat, wilcoxon_p_value = wilcoxon(differences)
                                logging.info(
                                    f"Layer {layer}, Pooling {pooling_method}: Wilcoxon test between {probe1} and {probe2}: stat={wilcoxon_stat:.4f}, p-value={wilcoxon_p_value:.4f}"
                                )
                            wilcoxon_results.append({
                                'Layer': layer,
                                'Pooling Method': pooling_method,
                                'Probe 1': probe1,
                                'Probe 2': probe2,
                                'stat': wilcoxon_stat,
                                'p-value': wilcoxon_p_value,
                            })
                        except Exception as e:
                            logging.error(
                                f"Error performing Wilcoxon test between {probe1} and {probe2} at layer {layer}, pooling {pooling_method}: {e}"
                            )
    except Exception as e:
        print(f"Error {e}")


    # Save statistical test results to CSV files
    t_test_df = pd.DataFrame(t_test_results)
    wilcoxon_df = pd.DataFrame(wilcoxon_results)

    t_test_csv_path = os.path.join(evaluation_dir, 'paired_t_test_results.csv')
    wilcoxon_csv_path = os.path.join(evaluation_dir, 'wilcoxon_test_results.csv')

    t_test_df.to_csv(t_test_csv_path, index=False)
    wilcoxon_df.to_csv(wilcoxon_csv_path, index=False)

    logging.info(f"Saved paired t-test results to '{t_test_csv_path}'.")
    logging.info(f"Saved Wilcoxon test results to '{wilcoxon_csv_path}'.")
    print(f"Saved statistical test results to '{evaluation_dir}'.")
    print("Statistical tests completed.")


if __name__ == "__main__":
    # Run evaluation based on predictions
    try:
        for model in ['meta-llama/Llama-3.1-8B-Instruct',
                      'meta-llama/Llama-3.2-1B-Instruct',
                      'meta-llama/Llama-3.2-3B-Instruct']:
                      # ['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-1B',
                     # 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']:
            model_name = model.replace('/', '_')
            for dataset_name in ['reduced_emotion', 'reduced_imdb', 'rotten_tomatoes', 'sst2']:
                print(f"Processing dataset: {dataset_name}")

                _, _, _, labels_test = load_sentiment_data(dataset_name)
                unique_labels = set(labels_test)
                num_classes = len(unique_labels)

                # Check if we need to run multi-class evaluation
                if num_classes is not None and num_classes > 2:
                    multi_class = True
                    logging.info(f"Running multi-class evaluation for {dataset_name}.")
                    print(f"Running multi-class evaluation for {dataset_name}.")
                else:
                    multi_class = False
                    logging.info(f"Running binary classification for {dataset_name}.")
                    print(f"Running binary classification for {dataset_name}.")

                # Evaluate Probe Predictions
                predictions_dir = os.path.join('predictions', model_name, dataset_name)
                if not os.path.exists(predictions_dir):
                    logging.warning(f"Prediction directory does not exist: {predictions_dir}")
                    print(f"Warning: Prediction directory does not exist: {predictions_dir}")
                    continue  # Skip this iteration
                evaluate_predictions(model_name, predictions_dir, labels_test, multi_class=multi_class)

                # Evaluate Prompt-Based Predictions
                # predictions_dir = os.path.join('predictions', model_name, 'Prompt_LLama', dataset_name)
                # if not os.path.exists(predictions_dir):
                #     logging.warning(f"Prediction directory does not exist: {predictions_dir}")
                #     print(f"Warning: Prediction directory does not exist: {predictions_dir}")
                #     continue  # Skip this iteration
                #
                # print(f"Evaluating prompt-based predictions in directory: {predictions_dir}")
                # evaluate_predictions_prompt(model_name, predictions_dir, labels_test, multi_class=multi_class)

    except Exception as e:
        logging.error(f"Error during final evaluation: {e}")
        print(f"Error during final evaluation: {e}")
