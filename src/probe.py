import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import numpy as np
from huggingface_hub import login, snapshot_download
import os
from typing import List
import gc
from tqdm import tqdm
from accelerate import Accelerator


class SentimentProbingToolkit:
    def __init__(
        self,
        model_name: str,
        on_gpu: bool,
        quantize: bool,
        quantization_mode: str,
        checkpoint_path: str = None,
        execution_mode: str = 'single_gpu',
    ):
        """
        Initialize the SentimentProbingToolkit.

        Args:
            model_name (str): Name of the model.
            on_gpu (bool): Whether to use GPU.
            checkpoint_path (str): Path to save checkpoints.
            execution_mode (str): Execution mode ('single_gpu', 'multi_gpu_single_node', 'multi_gpu_multi_node')
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.execution_mode = execution_mode
        self.on_gpu = on_gpu
        self.quantize = quantize
        self.quantization_mode = quantization_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() and on_gpu else 'cpu')
        self.load_model_and_tokenizer()

    def hf_login(self) -> None:
        model_dir = os.path.join('models', self.model_name.replace('/', '_'))
        token_file = 'hf_token.txt'
        if not os.path.exists(model_dir):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                login(token=token)
                logging.info("Successfully logged into Hugging Face.")
            except FileNotFoundError:
                logging.error(f"Token file '{token_file}' not found.")
            except Exception as e:
                logging.error(f"Error during Hugging Face login: {e}")

    def download_model(self, model_dir, specific_file='None'):
        """
        Downloads the model from Hugging Face Hub using snapshot_download.
        """
        logging.info(f"Starting download of model '{self.model_name}'...")
        try:
            snapshot_download(repo_id=self.model_name, local_dir=model_dir)
            logging.info(f"Model '{self.model_name}' successfully downloaded to '{model_dir}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{self.model_name}': {e}")

    # Function to determine the appropriate torch_dtype based on homogeneous GPU support
    def get_supported_torch_dtype(self):
        if torch.cuda.is_available():
            all_devices_capability_8 = True
            all_devices_capability_7 = True
            # Loop through all available GPUs
            for device_id in range(torch.cuda.device_count()):
                device_capability = torch.cuda.get_device_capability(device_id)
                # Check if any device has compute capability lower than 7.0
                if device_capability[0] < 7:
                    return torch.float32  # Default to float32 if any device has compute capability < 7.0
                # Check if all devices support capability 8.0+
                if device_capability[0] < 8:
                    all_devices_capability_8 = False  # Not all devices are >= 8.0
                # Check if all devices support capability 7.0+
                if device_capability[0] < 7:
                    all_devices_capability_7 = False  # Not all devices are >= 7.0
            # If all devices support compute capability 8.0+, return bfloat16
            if all_devices_capability_8:
                return torch.bfloat16
            # If all devices support compute capability 7.0+ (but not all 8.0+), return float16
            if all_devices_capability_7:
                return torch.float16
        return torch.float32  # Default to float32 if no suitable GPUs are available

    # Check if the model supports the determined torch_dtype
    def get_model_supported_dtype(self, model_dir, gpu_supported_dtype):
        config = AutoConfig.from_pretrained(model_dir)

        # List the supported dtypes for the model
        supported_dtypes = set()

        # By default, most models support float32, so add that
        supported_dtypes.add(torch.float32)

        # Check if model supports float16
        if getattr(config, "torch_dtype", None) == torch.float16:
            supported_dtypes.add(torch.float16)

        # Check if model supports bfloat16
        if getattr(config, "torch_dtype", None) == torch.bfloat16:
            supported_dtypes.add(torch.bfloat16)

        # Use the intersection of model-supported and GPU-supported dtypes
        if gpu_supported_dtype in supported_dtypes:
            return gpu_supported_dtype
        else:
            return torch.float32  # Default to float32 if no compatible dtype is found

    def load_quantized_model(self):
        """
        Load the quantized version of the LLM using llama_cpp.
        """
        model_dir = os.path.join('models', self.model_name.replace('/', '_'))

        # Download and save the model if not already done
        if not os.path.exists(model_dir):
            self.hf_login()
            os.makedirs(model_dir, exist_ok=True)
            self.download_model(model_dir, self.quantized_filename)

        try:
            logging.info(f"Loading quantized model '{self.model_name}'...")
            # self.model = Llama(
            #     model_path=f"{model_dir}/{self.quantized_filename}",
            #     embeddings=True,
            # )
            # Dynamically get the supported dtype based on the local GPU capabilities
            gpu_supported_dtype = self.get_supported_torch_dtype()
            # Ensure the dtype is compatible with the model
            # final_torch_dtype = self.get_model_supported_dtype(model_dir, gpu_supported_dtype)
            logging.info(f"Using {gpu_supported_dtype} precision")

            # Dynamically get the maximum memory for each GPU
            max_memory = {
                i: f"{torch.cuda.get_device_properties(i).total_memory * 0.9 / (1024 ** 3):.2f}GB"
                for i in range(torch.cuda.device_count())
            }

            # self.model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            # gguf_file = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                           gguf_file=self.quantized_filename)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                gguf_file=self.quantized_filename,
                output_hidden_states=True,
                torch_dtype=gpu_supported_dtype,
                device_map='auto',
                max_memory=max_memory,
                offload_folder='offload',
                trust_remote_code=True,
            )
            logging.info("Quantized model successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load quantized model '{self.model_name}': {e}")


    def load_model_and_tokenizer(self) -> None:
        """
        Load the tokenizer and model based on the execution mode, with optional quantization.
        """
        model_dir = os.path.join('models', self.model_name.replace('/', '_'))

        # Download and save the model if not already done
        if not os.path.exists(model_dir):
            self.hf_login()
            os.makedirs(model_dir, exist_ok=True)
            self.download_model(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Handle padding token if necessary
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dynamically get the supported dtype based on the local GPU capabilities
        gpu_supported_dtype = self.get_supported_torch_dtype()
        # Ensure the dtype is compatible with the model
        # final_torch_dtype = self.get_model_supported_dtype(model_dir, gpu_supported_dtype)
        logging.info(f"Using {gpu_supported_dtype} precision")

        # Dynamically get the maximum memory for each GPU
        max_memory = {
            i: f"{torch.cuda.get_device_properties(i).total_memory * 0.9 / (1024 ** 3):.2f}GB"
            for i in range(torch.cuda.device_count())
        }

        if self.quantize and self.quantization_mode == '8-bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantize and self.quantization_mode == '4-bit':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            quantization_config = None

        # Load model based on execution mode
        if self.execution_mode == 'single_gpu':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                output_hidden_states=True,
                torch_dtype=gpu_supported_dtype,
                device_map=None,
                trust_remote_code=True,
                quantization_config=quantization_config,
            ).to(self.device)

        # Load model on Multiple GPU
        elif self.execution_mode == 'multi_gpu_single_node':
            # Simplify model loading using device_map="auto"
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

        # Load model on Multiple GPU on Multiple Node
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

        logging.info(f"Model {self.model_name} loaded in mode {self.execution_mode}, occupied a total of {self.calculate_model_size():.2f} GB.")

        self.model.eval()

    def calculate_model_size(self):
        total_size = 0  # in bytes
        for param in self.model.parameters():
            num_elements = param.numel()  # Total number of elements
            dtype_size = param.element_size()  # Size in bytes for each element
            total_size += num_elements * dtype_size

        total_size_in_gb = total_size / (1024 ** 3)  # Convert bytes to GB
        return total_size_in_gb

    def get_total_layers(self) -> int:
        """
        Get the total number of layers in the model.

        Returns:
            int: Total number of layers.
        """
        if hasattr(self.model, 'encoder'):
            total_layers = len(self.model.encoder.layer)
        else:
            total_layers = self.model.config.num_hidden_layers
        return total_layers

    def get_hidden_states(
            self,
            texts: List[str],
            layer_idx: int = None,
            batch_size: int = 32,
            pooling_method: str = 'mean',
    ) -> np.ndarray:
        """
        Generate hidden states for the given texts at a specified layer.

        Args:
            texts (List[str]): List of input texts.
            layer_idx (int): Index of the layer to extract hidden states from.
            batch_size (int): Initial batch size for processing.
            pooling_method (str): Pooling method to use.

        Returns:
            np.ndarray: Array of hidden states.
        """
        # Handle multi-GPU, multi-node execution
        if self.execution_mode == 'multi_gpu_multi_node':
            # Split texts among processes for multi-node execution
            total_texts = len(texts)
            process_index = self.accelerator.process_index
            num_processes = self.accelerator.num_processes
            per_process = total_texts // num_processes
            start_idx = process_index * per_process
            # Ensure the last process handles the remaining texts
            end_idx = (process_index + 1) * per_process if process_index != num_processes - 1 else total_texts
            local_texts = texts[start_idx:end_idx]
        else:
            local_texts = texts  # Single-node execution uses all texts

        all_hidden_states = []
        # Initialize batch_size dynamically
        max_batch_size = batch_size
        min_batch_size = 1

        total_texts = len(local_texts)
        current_index = 0

        logging.info(
            f"Generating hidden states for layer {layer_idx} using {pooling_method} pooling, starting batch size: {max_batch_size}"
        )

        # Start with the maximum batch size and adjust dynamically
        current_batch_size = max_batch_size
        success = True

        # Initialize the tqdm progress bar
        with tqdm(total=total_texts, desc="Processing texts", unit="text") as pbar:
            while success and current_batch_size >= min_batch_size:
                while current_index < total_texts:
                    try:
                        # Prepare a batch of texts
                        batch_texts = local_texts[current_index: current_index + current_batch_size]

                        # Tokenize inputs without moving them to any device
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors='pt',
                            truncation=True,
                            padding='max_length',
                            max_length=128,
                        )

                        # Move inputs to the model's embedding device
                        embedding_device = self.model.get_input_embeddings().weight.device
                        inputs = {k: v.to(embedding_device) for k, v in inputs.items()}

                        with torch.no_grad():
                            # Check if the model can use mixed precision (float16 or bfloat16), otherwise use float32
                            use_mixed_precision = True # Assume mixed precision can be used

                            if self.on_gpu and torch.cuda.is_available():
                                # Loop through all available GPUs and check if any supports mixed precision
                                for device_id in range(torch.cuda.device_count()):
                                    device_capability = torch.cuda.get_device_capability(device_id)
                                    if device_capability[0] < 7:  # Compute capability 7.0 or higher
                                        use_mixed_precision = False  # Mixed precision cannot be used
                                        break

                            use_mixed_precision = use_mixed_precision and (
                                    self.model.dtype in [torch.float16, torch.bfloat16])

                            if use_mixed_precision:
                                # Use autocast for safe mixed-precision operations if supported
                                with torch.cuda.amp.autocast(enabled=True, dtype=self.model.dtype):
                                    outputs = self.model(**inputs)
                            else:
                                # Fall back to float32 operations if mixed precision is not supported
                                outputs = self.model(**inputs)

                        hidden_states = outputs.hidden_states  # Obtain hidden states from all layers
                        layer_hidden_states = hidden_states[layer_idx]  # Extract the hidden states for the specified layer

                        # Apply the specified pooling method
                        if pooling_method == 'mean':
                            batch_hidden_states = layer_hidden_states.mean(dim=1)
                        elif pooling_method == 'last-token':
                            batch_hidden_states = layer_hidden_states[:, -1, :]
                        elif pooling_method == 'max':
                            batch_hidden_states, _ = layer_hidden_states.max(dim=1)
                        elif pooling_method == 'min':
                            batch_hidden_states, _ = layer_hidden_states.min(dim=1)
                        elif pooling_method == 'concat-mean-max-min':
                            mean_pooled = layer_hidden_states.mean(dim=1)
                            max_pooled, _ = layer_hidden_states.max(dim=1)
                            min_pooled, _ = layer_hidden_states.min(dim=1)
                            batch_hidden_states = torch.cat((mean_pooled, max_pooled, min_pooled), dim=1)
                        elif pooling_method == 'attention':
                            attention_scores = layer_hidden_states.mean(dim=-1)
                            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
                            weighted_sum = (layer_hidden_states * attention_weights).sum(dim=1)
                            batch_hidden_states = weighted_sum
                        elif pooling_method == 'trainable':
                            batch_hidden_states = layer_hidden_states
                        else:
                            raise ValueError(f"Unknown pooling method: {pooling_method}")

                        # Move hidden states to CPU and convert them to numpy
                        batch_hidden_states = batch_hidden_states.float().cpu().numpy()
                        all_hidden_states.append(batch_hidden_states)

                        # Clear GPU memory
                        del hidden_states, layer_hidden_states, batch_hidden_states, outputs, inputs
                        torch.cuda.empty_cache()
                        gc.collect()

                        # Move to the next batch
                        current_index += current_batch_size
                        pbar.update(current_batch_size)

                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logging.warning(
                                f"CUDA Out of Memory with batch size {current_batch_size}. Reducing batch size.")
                            torch.cuda.empty_cache()
                            gc.collect()
                            success = False
                            # Reduce batch size by half
                            previous_batch_size = current_batch_size
                            current_batch_size = max(current_batch_size // 2, min_batch_size)
                            if current_batch_size == previous_batch_size:
                                # Cannot reduce batch size further
                                logging.error(f"Minimum batch size {min_batch_size} reached. Cannot process further.")
                                raise e
                            else:
                                logging.info(f"Retrying with batch size {current_batch_size}.")
                        else:
                            logging.error(f"Runtime error during inference: {e}")
                            raise e
                if current_index >= total_texts:
                    break

        # Concatenate hidden states for the entire dataset
        if pooling_method == 'trainable':
            local_hidden_states = np.concatenate(all_hidden_states, axis=0)
        else:
            local_hidden_states = np.vstack(all_hidden_states)

        # Handle multi-GPU, multi-node final gathering
        if self.execution_mode == 'multi_gpu_multi_node':
            # Gather results from all processes
            gathered_hidden_states = self.accelerator.gather(local_hidden_states)
            # Only the main process returns the final result
            if self.accelerator.is_main_process:
                final_hidden_states = gathered_hidden_states[:len(texts)]
                return final_hidden_states
            else:
                return None
        else:
            return local_hidden_states

