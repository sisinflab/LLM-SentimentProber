import os
import time
import logging
import yaml
import torch
import torch.nn as nn
import psutil
import pandas as pd
import joblib

from huggingface_hub import login, snapshot_download
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from accelerate import Accelerator
from utils import load_sentiment_data  # your custom data-loading function

# -------------------------
# Utility / Helper Methods
# -------------------------
def load_yaml_config(config_path: str) -> dict:
    """
    Loads parameters from a YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def measure_efficiency_with_sklearn(
    model,
    tokenizer,
    test_prompts,
    personal_model,
    device="cuda",
    pooling_strategy="first_token"
):
    """
    Measures inference efficiency while:
      1) Running a forward pass on the truncated LLaMA model
      2) Extracting the final hidden states
      3) Feeding those vectors into the scikit-learn model for prediction

    Returns a dictionary of metrics:
      - Total Params
      - Peak GPU Memory (MB)
      - Total Inference Time (s)
      - Avg Time/Sample (s)
      - Throughput (samples/s)
      - CPU Mem Before/After (MB)
    """
    # Put the model in eval mode and on the correct device
    model.eval()
    model.to(device)

    # Clear GPU memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)

    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Record CPU memory usage before
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss  # in bytes

    start_time = time.perf_counter()

    # Function to forward pass the truncated model + scikit predict
    def forward_and_predict(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Final hidden layer: outputs.hidden_states[-1]
        # shape: [batch_size, seq_len, hidden_size]
        last_hidden = outputs.hidden_states[-1]

        if personal_model != None:
            # Apply the specified pooling method
            if pooling_strategy == 'mean':
                vector = last_hidden.mean(dim=1)
            elif pooling_strategy == 'last-token':
                vector = last_hidden[:, -1, :]
            elif pooling_strategy == 'max':
                vector, _ = last_hidden.max(dim=1)
            elif pooling_strategy == 'min':
                vector, _ = last_hidden.min(dim=1)
            elif pooling_strategy == 'concat-mean-max-min':
                mean_pooled = last_hidden.mean(dim=1)
                max_pooled, _ = last_hidden.max(dim=1)
                min_pooled, _ = last_hidden.min(dim=1)
                vector = torch.cat((mean_pooled, max_pooled, min_pooled), dim=1)
            elif pooling_strategy == 'attention':
                attention_scores = last_hidden.mean(dim=-1)
                attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
                weighted_sum = (last_hidden * attention_weights).sum(dim=1)
                vector = weighted_sum
            elif pooling_strategy == 'trainable':
                vector = last_hidden
            else:
                raise ValueError(f"Unknown pooling method: {pooling_strategy}")


            # Convert to numpy for scikit-learn
            vector_np = vector.cpu().numpy()

            # scikit-learn logistic regression inference
            pred = personal_model.predict(vector_np)
            return pred
        else:
            return None

    # Run inference on all prompts
    for prompt in test_prompts:
        _ = forward_and_predict(prompt)

    end_time = time.perf_counter()
    cpu_mem_after = process.memory_info().rss

    peak_memory_allocated = torch.cuda.max_memory_allocated(device=device)

    total_inference_time = end_time - start_time
    avg_time_per_sample = total_inference_time / len(test_prompts) if test_prompts else float('nan')
    throughput = len(test_prompts) / total_inference_time if total_inference_time > 0 else float('nan')

    metrics = {
        "Total Params": total_params,
        "Peak GPU Mem (MB)": peak_memory_allocated / (1024 ** 2),
        "Total Inference Time (s)": total_inference_time,
        "Avg Time/Sample (s)": avg_time_per_sample,
        "Throughput (samples/s)": throughput,
        "CPU Mem Before (MB)": cpu_mem_before / (1024**2),
        "CPU Mem After (MB)": cpu_mem_after / (1024**2)
    }
    return metrics

def format_metrics_for_print(raw_metrics: dict) -> dict:
    """
    Convert raw metrics into more human-readable strings (especially timing in seconds).
    """
    fm = {}

    # Example: total params as integer with commas
    total_params = raw_metrics["Total Params"]
    fm["Total Params"] = f"{int(round(total_params)):,}"

    # Example: peak GPU mem
    peak_gpu = raw_metrics["Peak GPU Mem (MB)"]
    fm["Peak GPU Mem (MB)"] = f"{int(round(peak_gpu))} MB"

    # 1) Convert 'Total Inference Time (s)' to a nice decimal in seconds
    total_inf_s = raw_metrics["Total Inference Time (s)"]  # e.g. 1.010622e+01
    fm["Total Inference Time"] = f"{total_inf_s:.2f} s"    # e.g. "10.11 s"

    # 2) Example: avg time/sample in ms
    avg_time_s = raw_metrics["Avg Time/Sample (s)"]
    fm["Avg Time/Sample"] = f"{avg_time_s * 1000:.2f} ms"

    # Throughput
    throughput = raw_metrics["Throughput (samples/s)"]
    fm["Throughput (samples/s)"] = f"{int(round(throughput))}"

    # CPU Mem
    cpu_before = raw_metrics["CPU Mem Before (MB)"]
    fm["CPU Mem Before (MB)"] = f"{int(round(cpu_before))} MB"
    cpu_after = raw_metrics["CPU Mem After (MB)"]
    fm["CPU Mem After (MB)"] = f"{int(round(cpu_after))} MB"

    # If FLOPs exist, convert to a more readable form
    if "FLOPs (1 pass)" in raw_metrics:
        flops = raw_metrics["FLOPs (1 pass)"]
        fm["FLOPs (1 pass)"] = f"{int(round(flops)):,}"

    return fm

# -------------------------
# Class with your Methods
# -------------------------
class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.execution_mode = config["execution_mode"]
        self.quantize = config["quantize"]
        self.quantization_mode = config["quantization_mode"]
        self.device = config["device"]
        self.cutoff = config["cutoff"]
        self.personal_model_path = config["personal_model_path"]
        self.dataset_name = config.get("dataset_name")
        self.pooling_strategy = config.get("pooling_strategy")

        self.model = None
        self.tokenizer = None
        self.accelerator = None
        self.personal_model = None

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def hf_login(self) -> None:
        """
        Login to Hugging Face using token stored in hf_token.txt
        """
        token_file = 'hf_token.txt'
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
            login(token=token)
            logging.info("Successfully logged into Hugging Face.")
        except FileNotFoundError:
            logging.error(f"Token file '{token_file}' not found. Proceeding without HF login.")
        except Exception as e:
            logging.error(f"Error during Hugging Face login: {e}")

    def download_model(self, model_dir):
        """
        Download model files from Hugging Face Hub to local directory.
        """
        logging.info(f"Downloading {self.model_name} to {model_dir} ...")
        snapshot_download(repo_id=self.model_name, local_dir=model_dir)
        logging.info(f"Model {self.model_name} downloaded successfully.")

    def get_supported_torch_dtype(self):
        """
        Determines a suitable torch dtype for GPU (float16, bfloat16, etc.).
        For simplicity, we return torch.float16 if CUDA is available; otherwise float32.
        """
        if torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32

    def calculate_model_size(self):
        """
        Returns an approximate model size in GB, based on param count & dtype.
        """
        if self.model is None:
            return 0
        param_count = sum(p.numel() for p in self.model.parameters())
        dtype_itemsize = 2 if self.model.dtype == torch.float16 else 4
        size_gb = (param_count * dtype_itemsize) / (1024 ** 3)
        return size_gb

    def load_model_and_tokenizer(self) -> None:
        """
        Load tokenizer and model from local dir or Hugging Face Hub,
        respecting execution mode and quantization options.
        """
        model_dir = os.path.join('models', self.model_name.replace('/', '_'))
        if not os.path.exists(model_dir):
            self.hf_login()
            os.makedirs(model_dir, exist_ok=True)
            self.download_model(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        gpu_supported_dtype = self.get_supported_torch_dtype()
        logging.info(f"Using {gpu_supported_dtype} precision")

        # For multi-GPU memory mapping
        max_memory = {
            i: f"{torch.cuda.get_device_properties(i).total_memory * 0.9 / (1024 ** 3):.2f}GB"
            for i in range(torch.cuda.device_count())
        }

        # Handle quantization
        quantization_config = None
        if self.quantize:
            if self.quantization_mode == '8-bit':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.quantization_mode == '4-bit':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # --- Single GPU ---
        if self.execution_mode == 'single_gpu':
            self.model = AutoModel.from_pretrained(
                model_dir,
                output_hidden_states=True,
                torch_dtype=gpu_supported_dtype,
                device_map=None,
                trust_remote_code=True,
                quantization_config=quantization_config,
            ).to(self.device)

        # --- Multi GPU Single Node ---
        elif self.execution_mode == 'multi_gpu_single_node':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                output_hidden_states=True,
                torch_dtype=gpu_supported_dtype,
                device_map='auto',
                max_memory=max_memory,
                offload_folder='offload',
                trust_remote_code=True,
                quantization_config=quantization_config,
            )

        # --- Multi GPU Multi Node ---
        elif self.execution_mode == 'multi_gpu_multi_node':
            # Use 'accelerate' to manage multi-node multi-GPU execution
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                output_hidden_states=True,
                torch_dtype=gpu_supported_dtype,
                device_map='auto',
                offload_folder='offload',
                offload_state_dict=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

        logging.info(
            f"Model {self.model_name} loaded in mode {self.execution_mode}, "
            f"occupied a total of {self.calculate_model_size():.2f} GB."
        )
        self.model.eval()

    def slice_model_layers(self):
        """
        Remove top layers beyond a given cutoff, if applicable (LLaMA has model.model.layers).
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            logging.warning("Model does not have a 'model.layers' structure to slice.")
            return

        original_num_layers = len(self.model.model.layers)
        if self.cutoff < original_num_layers:
            self.model.model.layers = self.model.model.layers[:self.cutoff]
            logging.info(
                f"Sliced the model to keep only the first {self.cutoff} layers "
                f"(original: {original_num_layers}, now: {len(self.model.model.layers)})."
            )
        else:
            logging.info(
                f"Cutoff {self.cutoff} >= model layers ({original_num_layers}). No slicing performed."
            )

    def load_sklearn_model(self):
        """
        Load the scikit-learn model (e.g., logistic regression) from disk.
        """
        if not os.path.exists(self.personal_model_path):
            logging.warning(f"Personal model file {self.personal_model_path} not found. Skipping.")
            self.personal_model_path = None
            return None
        logging.info(f"Loading scikit-learn model from {self.personal_model_path}")
        return joblib.load(self.personal_model_path)

    def run_inference_and_measure_efficiency(self):
        """
        1. Loads a sentiment test set from load_sentiment_data(self.dataset_name).
        2. Loads the scikit-learn model (personal_model_path).
        3. Truncates LLaMA, obtains last hidden states, feeds them to scikit model.
        4. Measures and prints a table of efficiency metrics.
        """
        # Load test portion from your custom sentiment dataset
        _, _, texts_test, labels_test = load_sentiment_data(self.dataset_name)
        if not texts_test:
            logging.info("No test prompts found in dataset. Skipping efficiency measurement.")
            return

        # Load scikit model
        self.personal_model = self.load_sklearn_model()
        if self.personal_model is None:
            logging.warning("No scikit-learn model loaded. The final step won't run classification.")
            logging.warning("Still measuring time/memory for LLaMA forward pass only.")

        # Measure and print metrics
        raw_metrics = measure_efficiency_with_sklearn(
            model=self.model,
            tokenizer=self.tokenizer,
            test_prompts=texts_test,
            personal_model=self.personal_model,
            device=self.device,
            pooling_strategy=self.pooling_strategy  # or "mean_pool", etc.
        )

        # Convert them to friendly strings
        formatted = format_metrics_for_print(raw_metrics)

        # Display as a Pandas DataFrame
        df = pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"])
        print("\n===== Efficiency Metrics =====")
        print(df.to_string(index=False))


def main():
    # 1. Load config from YAML
    config_path = "sentri_llama_config.yaml"  # Modify path if needed
    config = load_yaml_config(config_path)

    # 2. Initialize ModelManager
    manager = ModelManager(config)

    # 3. Load Model & Tokenizer
    manager.load_model_and_tokenizer()

    # 4. Slice Model Layers if cutoff < total
    #manager.slice_model_layers()

    # 5. Run inference and measure efficiency
    manager.run_inference_and_measure_efficiency()


if __name__ == "__main__":
    main()
