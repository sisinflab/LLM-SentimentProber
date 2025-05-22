from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login, snapshot_download
from accelerate import Accelerator
from utils import load_sentiment_data
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
import logging


class LLamaRunner:
    def __init__(self, model_name="huggingface/llama-3.2", execution_mode="single_gpu", device=None):
        """
        Initializes the LLamaRunner class for loading and running different LLaMA models.

        Args:
            model_name (str): The Hugging Face model identifier for LLaMA.
            execution_mode (str): Execution mode for the model ("single_gpu", "multi_gpu_single_node", "multi_gpu_multi_node").
            device (str or None): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
        """
        self.model_name = model_name
        self.execution_mode = execution_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model_dir = os.path.join('models', model_name.replace('/', '_'))
        self.tokenizer, self.model = self.load_model_and_tokenizer()

        # Set up the text generation pipeline
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def hf_login(self):
        """Login to Hugging Face if needed to download private models."""
        token_file = 'hf_token.txt'
        if not os.path.exists(self.model_dir):
            try:
                with open(token_file, 'r') as f:
                    login(token=f.read().strip())
                    logging.info("Logged into Hugging Face successfully.")
            except FileNotFoundError:
                logging.error("Token file not found. Ensure 'hf_token.txt' exists.")

    def load_model_and_tokenizer(self):
        """Load or download model and tokenizer based on the specified execution mode."""
        if not os.path.exists(self.model_dir):
            self.hf_login()
            snapshot_download(repo_id=self.model_name, local_dir=self.model_dir)
            logging.info(f"Model '{self.model_name}' downloaded to '{self.model_dir}'.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        dtype = self.get_supported_torch_dtype()
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory * 0.9 / (1024 ** 3):.2f}GB"
                      for i in range(torch.cuda.device_count())}

        if self.execution_mode == "single_gpu":
            model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=dtype).to(self.device)
        elif self.execution_mode == "multi_gpu_single_node":
            model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto", torch_dtype=dtype,
                                                         max_memory=max_memory)
        elif self.execution_mode == "multi_gpu_multi_node":
            self.accelerator = Accelerator()
            model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto", torch_dtype=dtype)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

        logging.info(f"Model {self.model_name} loaded in mode {self.execution_mode}.")
        return tokenizer, model.eval()

    @staticmethod
    def get_supported_torch_dtype():
        """Determine supported torch dtype based on GPU capabilities."""
        if torch.cuda.is_available():
            capability_levels = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
            if all(level >= 8 for level in capability_levels):
                return torch.bfloat16
            elif all(level >= 7 for level in capability_levels):
                return torch.float16
        return torch.float32

    def generate(self, prompt, max_new_tokens=1, num_return_sequences=1):
        """
        Generates text based on a given prompt.

        Args:
            prompt (str): Input text prompt.
            max_length (int): Maximum length of the generated text.
            num_return_sequences (int): Number of generated sequences to return.

        Returns:
            List of generated text sequences.
        """
        return self.pipeline(prompt, do_sample=False,  max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)

def save_predictions_tsv(model_name, dataset_name, y_pred, prompt_type: str) -> None:
    """
    Save predictions to a TSV file for statistical analysis.

    Args:
        y_pred (np.ndarray): Predictions to save.
        prompt_type (str): Prompt Type.
    """
    # Define the directory path using the dataset name and model name
    dir_path = os.path.join(
        'predictions',
        model_name.replace('/', '_'),
        "Prompt_LLama",
        dataset_name
    )

    # Create the filename and full file path
    filename = f"{prompt_type}.tsv"
    filepath = os.path.join(dir_path, filename)

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Create a DataFrame and save to TSV
    df = pd.DataFrame({'y_pred': y_pred})
    df.to_csv(filepath, sep='\t', index=False)

    logging.info(f"Saved predictions to {filepath}")

def zero_shot_prompt(text, task_type="binary"):
    """
    Generates a Zero-Shot prompt for sentiment or emotion analysis.
    The prompt is structured to strongly encourage the model to respond
    with only a numeric value (0 or 1 for binary sentiment; 0-5 for emotions),
    and no additional commentary or explanation.

    Args:
        text (str): The text to classify.
        task_type (str): "binary" for sentiment polarity (positive/negative)
                         or "emotion" for emotion classification.

    Returns:
        list: Zero-Shot prompt messages for the specified classification task.
    """
    # Base system instruction: the model should not deviate.
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant trained to perform strict sentiment and emotion classification. "
                "You MUST only respond with the numeric label corresponding to the classification. "
                "Do not provide any explanations, reasoning, or any text other than the required numeric value."
            )
        },
        {
            "role": "user",
            "content": f"Classify the sentiment of the following text: '{text}'"
        }
    ]

    # Add task-specific instructions as an assistant message.
    if task_type == "binary":
        messages.append({
            "role": "assistant",
            "content": (
                "If the sentiment is positive, respond with '1'. "
                "If the sentiment is negative, respond with '0'. "
                "No other text, explanation, or formatting."
            )
        })
    elif task_type == "emotion":
        messages.append({
            "role": "assistant",
            "content": (
                "Classify the text into one of the following emotions and respond only with the corresponding number: "
                "0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise. "
                "No explanation or additional text."
            )
        })

    return messages

def few_shot_prompt(text, task_type="binary"):
    """
    Generates a Few-Shot prompt for sentiment or emotion analysis with examples.
    This version ensures the model responds only with the specified numeric value,
    providing no explanations or extra commentary.

    Args:
        text (str): The text to classify.
        task_type (str): "binary" for sentiment polarity (positive/negative)
                         or "emotion" for emotion classification.

    Returns:
        list: Few-Shot prompt messages for the specified classification task.
    """
    if task_type == "binary":
        examples = (
            "Examples:\n"
            "'I love this product!' => 1\n"
            "'I am disappointed with the service.' => 0\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant trained for sentiment and emotion analysis. "
                    "You MUST only respond with the correct numeric label. "
                    "Do not provide explanations or any additional text."
                )
            },
            {
                "role": "user",
                "content": f"{examples}\nClassify the following text sentiment:\n'{text}'"
            },
            {
                "role": "assistant",
                "content": (
                    "If the sentiment is positive, respond with '1'. "
                    "If the sentiment is negative, respond with '0'. "
                    "No other text, explanation, or formatting."
                )
            }
        ]

    elif task_type == "emotion":
        examples = (
            "Examples:\n"
            "'This is the worst day of my life.' => 0\n"
            "'I feel so joyful and alive!' => 1\n"
            "'I feel so deeply connected and grateful for you in my life.' => 2\n"
            "'I am so angry right now.' => 3\n"
            "'I’m really scared and worried about what might happen next.' => 4\n"
            "'Wow, I didn't expect that at all! This is completely unexpected!' => 5\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant trained for sentiment and emotion analysis. "
                    "You MUST only respond with the correct numeric label. "
                    "Do not provide explanations or any additional text."
                )
            },
            {
                "role": "user",
                "content": f"{examples}\nClassify the following text emotion:\n'{text}'"
            },
            {
                "role": "assistant",
                "content": (
                    "Respond only with one of the following numbers: "
                    "0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise. "
                    "No other text, explanation, or formatting."
                )
            }
        ]

    return messages

def chain_of_thought_prompt(text, task_type="binary"):
    """
    Generates a Chain-of-Thought prompt for sentiment or emotion analysis.
    This version instructs the model to reason step-by-step internally (chain-of-thought)
    and then provide only the final numeric answer without any explanation.
    The chain-of-thought reasoning is encouraged, but not to be included in the final output.

    Args:
        text (str): The text to classify.
        task_type (str): "binary" for sentiment polarity (positive/negative)
                         or "emotion" for emotion classification.

    Returns:
        list: Chain-of-Thought prompt messages for the specified classification task.
    """
    if task_type == "binary":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in sentiment and emotion analysis. "
                    "Think step-by-step through the reasoning process (chain-of-thought) privately, "
                    "but provide only the final numeric classification as instructed. "
                    "Do not include reasoning steps in the output."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the sentiment of the following text:\n\n'{text}'\n\n"
                    "Carefully reason step-by-step to determine the sentiment. "
                    "Output only '1' for positive sentiment or '0' for negative sentiment as your final response."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "I will reason step-by-step internally to determine the sentiment. "
                    "However, my final response will be '1' for positive sentiment or '0' for negative sentiment, "
                    "with no explanation included in the output."
                )
            }
        ]

    elif task_type == "emotion":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in sentiment and emotion analysis. "
                    "Think step-by-step through the reasoning process (chain-of-thought) privately, "
                    "but provide only the final numeric classification as instructed. "
                    "Do not include reasoning steps in the output."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the emotion of the following text:\n\n'{text}'\n\n"
                    "Carefully reason step-by-step to identify the best-matching emotion. "
                    "Output only the corresponding number as your final response:\n"
                    "0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "I will reason step-by-step internally to determine the most appropriate emotion. "
                    "My final response will be one of the following numbers:\n"
                    "0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise. "
                    "No reasoning will be included in the output."
                )
            }
        ]

    else:
        raise ValueError("Invalid task_type. Use 'binary' for sentiment analysis or 'emotion' for emotion classification.")

    return messages

def classify_output(llm_output, task_type="binary"):
    """
    Maps the LLM output to a classification label based on the specified task.

    Args:
        llm_output (str): The raw output from the LLM.
        task_type (str): "binary" for sentiment polarity or "emotion" for emotion classification.

    Returns:
        int: Classification label.
    """
    if task_type == "binary":
        return 1 if "1" in llm_output else 0
    elif task_type == "emotion":
        emotion_mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        return emotion_mapping.get(llm_output.strip(), -1)  # Default to -1 if unrecognized

# Example usage
if __name__ == "__main__":
    test_on_reduced_dataset = False

    for model_name in ['meta-llama/Llama-3.1-8B-Instruct',
                       'meta-llama/Llama-3.2-1B-Instruct',
                       'meta-llama/Llama-3.2-3B-Instruct']:

        runner = LLamaRunner(model_name=model_name, execution_mode="multi_gpu_single_node")

        for dataset_name in ['reduced_emotion', 'reduced_imdb', 'rotten_tomatoes', 'sst2']:
            logging.info(f"Starting experiment with {model_name} on {dataset_name}.")

            # Load data and handle exceptions
            try:
                _, _, texts_test, labels_test = load_sentiment_data(dataset_name)
                if test_on_reduced_dataset:
                    texts_test = texts_test[:100]
                    labels_test = labels_test[:100]
            except Exception as e:
                logging.error(f"Error loading data for dataset {dataset_name}: {e}")
                continue

            # Determine the number of classes based on the labels
            unique_labels = set(labels_test)
            num_classes = len(unique_labels)
            logging.info(f"Number of classes in dataset '{dataset_name}': {num_classes}")

            # Check if we need to run multi-class inference
            if num_classes is not None and num_classes > 2:
                multi_class = True
                task_type = 'emotion'
            else:
                multi_class = False
                task_type = "binary"

            for prompt_engineer_type in ['CoT', 'CoT2']:# ['Zero-Shot', 'Few-Shot', 'CoT']:
                all_preds = []
                logging.info(f"Starting experiments using {prompt_engineer_type} techniques.\n")

                for text_test in tqdm(texts_test, desc=f"{prompt_engineer_type} - {model_name} on {dataset_name}"):
                    if prompt_engineer_type == 'Zero-Shot':
                        prompt = zero_shot_prompt(text_test, task_type)
                    elif prompt_engineer_type == 'Few-Shot':
                        prompt = few_shot_prompt(text_test, task_type)
                    elif prompt_engineer_type == 'CoT':
                        prompt = chain_of_thought_prompt(text_test, task_type)
                    else:
                        logging.info("ERROR: No Prompt Engineering Techniques Recognized.")
                        break

                    llm_output = runner.generate(prompt)
                    label = classify_output(llm_output[0]['generated_text'][-1]['content'], task_type)
                    all_preds.append(label)

                save_predictions_tsv(model_name, dataset_name, all_preds, prompt_engineer_type)