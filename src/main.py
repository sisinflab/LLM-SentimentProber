import logging
import concurrent.futures
import yaml
import argparse
import time
import warnings
import os
import multiprocessing

from probe import SentimentProbingToolkit
from model_trainer import ModelTrainer
from utils import (
    load_sentiment_data,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    concatenate_csv_files,
)
from evaluation import evaluate_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log', mode='w'),
        logging.StreamHandler(),
    ],
    force=True,
)

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

def format_time(seconds: float) -> str:
    """
    Convert seconds to a formatted string of hours, minutes, and seconds.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string in the format 'Xh Ym Zs'.
    """
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {int(secs)}s"

def run_experiment(
    experiment: dict,
    test_on_reduced_dataset: bool,
    sequential: bool = False,
    token_level_exploration: bool = False,
    seed: int = 42,
    execution_mode: str = 'single_gpu',
    quantize: bool = False,
    quantization_mode: str = 'None',
) -> None:
    """
    Run a single experiment based on the provided configuration.

    Args:
        experiment (dict): Experiment configuration.
        test_on_reduced_dataset (bool): Whether to test on a reduced dataset.
        sequential (bool): Whether to run probes sequentially.
        token_level_exploration (bool): Whether to explore token-level pooling methods.
        seed (int): Random seed for reproducibility.
    """
    model_names = experiment['model_name']
    for model_name in model_names:
        dataset_names = experiment['dataset_name']
        probe_types = experiment['probe_types']
        checkpoint_path = os.path.join(experiment['checkpoint_path'], model_name.replace('/', '_'))
        n_trials = experiment.get('n_trials', 5)  # Default to 5 trials if not specified
        on_gpu = experiment['device'].lower() == 'gpu'
        batch_size = experiment.get('batch_size', 32)  # Configurable batch size

        # Define pooling methods based on the token_level_exploration flag
        if token_level_exploration:
            pooling_methods = [
                'mean',
                'last-token',
                'max',
                'min',
                'concat-mean-max-min',
                'attention',
                # 'trainable',  # Uncomment if using trainable pooling
            ]
        else:
            pooling_methods = ['mean']  # Default pooling method

        if dataset_names[0] == 'multilingual':
            dataset_names = ['arabic', 'english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish']

        for dataset_name in dataset_names:
            logging.info(f"Starting experiment with {model_name} on {dataset_name}.")

            # Load data and handle exceptions
            try:
                texts_train, labels_train, texts_test, labels_test = load_sentiment_data(dataset_name)
                if test_on_reduced_dataset:
                    texts_train = texts_train[:100]
                    labels_train = labels_train[:100]
                    texts_test = texts_test[:100]
                    labels_test = labels_test[:100]
            except Exception as e:
                logging.error(f"Error loading data for dataset {dataset_name}: {e}")
                continue  # Skip to the next dataset if data loading fails

            # Determine the number of classes based on the labels
            unique_labels = set(labels_train)
            num_classes = len(unique_labels)
            logging.info(f"Number of classes in dataset '{dataset_name}': {num_classes}")

            # Check if we need to run multi-class training
            if num_classes is not None and num_classes > 2:
                multi_class = True
                logging.info(f"Running multi-class training for {dataset_name}.")
            else:
                multi_class = False
                logging.info(f"Running binary classification for {dataset_name}.")

            dataset_checkpoint_path = os.path.join(checkpoint_path, dataset_name)
            probing_toolkit = SentimentProbingToolkit(
                model_name,
                on_gpu,
                quantize,
                quantization_mode,
                dataset_checkpoint_path,
                execution_mode=execution_mode,
            )

            # Determine the total number of layers in the model
            total_layers = probing_toolkit.get_total_layers()
            logging.info(f"Model total layers: {total_layers}")

            # Load checkpoint if it exists
            checkpoint = load_checkpoint(dataset_checkpoint_path)
            start_layer = checkpoint.get('last_layer', 0)

            for layer_idx in range(start_layer, total_layers):
                logging.info(f"Processing layer {layer_idx}.")

                for pooling_method in pooling_methods:
                    logging.info(f"Using pooling method: {pooling_method}")

                    # Check if all probes for this layer and pooling method are already completed
                    completed_probes = checkpoint.get('completed_probes', {})
                    layer_probes = completed_probes.get(str(layer_idx), {})
                    pooling_probes = layer_probes.get(pooling_method, [])

                    # If all probe types are completed, skip this pooling method
                    if set(probe_types).issubset(set(pooling_probes)):
                        logging.info(
                            f"All probes for layer {layer_idx} with pooling method {pooling_method} are already completed. Skipping."
                        )
                        continue

                    # Generate hidden states once per layer and pooling method
                    try:
                        logging.info(
                            f"Starting to generate hidden states for TRAINING data. "
                        )
                        hidden_states_train = probing_toolkit.get_hidden_states(
                            texts_train,
                            layer_idx=layer_idx,
                            pooling_method=pooling_method,
                            batch_size=batch_size,
                        )
                        logging.info(
                            f"Completed generating hidden states for TRAINING data. "
                        )
                        logging.info(
                            f"Starting to generate hidden states for TEST data. "
                        )
                        hidden_states_test = probing_toolkit.get_hidden_states(
                            texts_test,
                            layer_idx=layer_idx,
                            pooling_method=pooling_method,
                            batch_size=batch_size,
                        )
                        logging.info(
                            f"Completed generating hidden states for TEST data"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error generating hidden states for layer {layer_idx} with pooling {pooling_method}: {e}"
                        )
                        continue  # Skip to the next pooling method if hidden state generation fails

                    # Prepare arguments for ModelTrainer
                    trainer_args = {
                        'hidden_states_train': hidden_states_train,
                        'labels_train': labels_train,
                        'hidden_states_test': hidden_states_test,
                        'labels_test': labels_test,
                        'n_trials': n_trials,
                        'seed': seed,
                        'on_gpu': on_gpu,
                        'dataset_name': dataset_name,
                        'model_name': model_name,
                        'pooling_method': pooling_method,
                        'multi_class': multi_class,  # Pass multi_class flag
                    }

                    if sequential:
                        # Sequential execution of probe types
                        for probe_type in probe_types:
                            if probe_type in pooling_probes:
                                logging.info(
                                    f"Skipping {probe_type} at layer {layer_idx} with pooling method {pooling_method} (already completed)."
                                )
                                continue

                            logging.info(
                                f"Processing {probe_type} for layer {layer_idx} with pooling method {pooling_method}."
                            )

                            # Include pooling method and probe type in the result file name
                            result_file = os.path.join(
                                dataset_checkpoint_path,
                                f'layer_{layer_idx}_{pooling_method}_{probe_type}.csv',
                            )

                            classifier = ModelTrainer(
                                probe_type,
                                result_file=result_file,
                                **trainer_args,
                            )

                            try:
                                result = classifier.train_and_evaluate(layer_idx)
                                logging.info(
                                    f"{probe_type} at layer {layer_idx} with pooling method {pooling_method} completed with result: {result}"
                                )

                                # Update checkpoint
                                pooling_probes.append(probe_type)
                                layer_probes[pooling_method] = pooling_probes
                                completed_probes[str(layer_idx)] = layer_probes
                                checkpoint['completed_probes'] = completed_probes
                                save_checkpoint(dataset_checkpoint_path, checkpoint)

                            except Exception as exc:
                                logging.error(
                                    f"{probe_type} at layer {layer_idx} with pooling method {pooling_method} generated an exception: {exc}"
                                )

                    else:
                        # Parallel execution using ProcessPoolExecutor
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            futures = {}
                            for probe_type in probe_types:
                                if probe_type in pooling_probes:
                                    logging.info(
                                        f"Skipping {probe_type} at layer {layer_idx} with pooling method {pooling_method} (already completed)."
                                    )
                                    continue

                                logging.info(
                                    f"Submitting {probe_type} for layer {layer_idx} with pooling method {pooling_method} to executor."
                                )

                                result_file = os.path.join(
                                    dataset_checkpoint_path,
                                    f'layer_{layer_idx}_{pooling_method}_{probe_type}.csv',
                                )

                                classifier = ModelTrainer(
                                    probe_type,
                                    result_file=result_file,
                                    **trainer_args,
                                )

                                future = executor.submit(classifier.train_and_evaluate, layer_idx)
                                futures[future] = probe_type

                            # Collect results as they complete
                            for future in concurrent.futures.as_completed(futures):
                                probe_type = futures[future]
                                try:
                                    result = future.result()
                                    logging.info(
                                        f"{probe_type} at layer {layer_idx} with pooling method {pooling_method} completed with result: {result}"
                                    )

                                    # Update checkpoint
                                    pooling_probes.append(probe_type)
                                    layer_probes[pooling_method] = pooling_probes
                                    completed_probes[str(layer_idx)] = layer_probes
                                    checkpoint['completed_probes'] = completed_probes
                                    save_checkpoint(dataset_checkpoint_path, checkpoint)

                                except Exception as exc:
                                    logging.error(
                                        f"{probe_type} at layer {layer_idx} with pooling method {pooling_method} generated an exception: {exc}"
                                    )

                    # Save checkpoint after each pooling method
                    checkpoint['completed_probes'][str(layer_idx)] = layer_probes
                    save_checkpoint(dataset_checkpoint_path, checkpoint)

                # Update last processed layer after all pooling methods
                checkpoint['last_layer'] = layer_idx + 1
                save_checkpoint(dataset_checkpoint_path, checkpoint)

            logging.info(f"Completed experiment with {model_name} on {dataset_name}.")

            # Run final evaluation after all experiments
            try:
                y_true = labels_test  # Assuming labels_test is the ground truth
                predictions_dir = os.path.join('predictions', model_name.replace('/','_'), dataset_name)
                concatenate_csv_files(dataset_checkpoint_path,
                                      f"{model_name.replace('/', '_')}_probe_results.csv")
                evaluate_predictions(model_name, predictions_dir, y_true, multi_class=multi_class)
            except Exception as e:
                logging.error(f"Error during final evaluation: {e}")

def sequential_experiments(experiments: list, options: dict) -> None:
    """
    Run experiments sequentially based on the provided configuration.

    Args:
        experiments (list): List of experiment configurations.
        options (dict): Global options for experiments.
    """
    # Extract options from the configuration
    sequential = options.get('sequential', False)
    token_level_exploration = options.get('token_level_exploration', False)
    seed = options.get('seed', 42)
    test_on_reduced_dataset = options.get('test_on_reduced_dataset', False)
    execution_mode = options.get('execution_mode', 'single_gpu')
    quantize = options.get('quantized_model', False)
    quantization_mode = options.get('quantization_mode', None)

    for exp in experiments:
        run_experiment(
            exp,
            test_on_reduced_dataset,
            sequential=sequential,
            token_level_exploration=token_level_exploration,
            seed=seed,
            execution_mode=execution_mode,
            quantize=quantize,
            quantization_mode=quantization_mode,
        )

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        '--config-file',
        type=str,
        default='example_experiments.yaml',
        help='Path to the YAML configuration file for experiments',
    )

    args = parser.parse_args()

    # Read configuration from YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    options = config.get('options', {})
    experiments = config.get('experiment', [])

    # Set seed
    seed = options.get('seed', 42)
    set_seed(seed)  # Set seed for reproducibility

    start_time = time.time()  # Start timing

    sequential_experiments(experiments, options)

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    formatted_time = format_time(total_time)
    logging.info(f"Total time needed to complete all experiments: {formatted_time}")
