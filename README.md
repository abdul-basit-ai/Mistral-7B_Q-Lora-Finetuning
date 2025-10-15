# Mistral-7B_Q-Lora-Finetuning(Quantized-Lora finetuning)
In this project I finetuned Miistral 7-B instruct model(mistral-7b-instruct-v0.2-bnb-4bit) on ALPHA-cleaned dataset (only on 10% of it). 
Weight and Biases Report : https://api.wandb.ai/links/basitmal36-university-of-paris-saclay/519cxzlq

The fine-tuning process uses several specialized configurations for maximum efficiency. The base model used is unsloth/mistral-7b-instruct-v0.2-bnb-4bit, which is a 4-bit quantized version of Mistral 7B, directly enabling QLoRA. This quantization is controlled by setting load_in_4bit = True, which results in massive memory savings essential for training large models on consumer GPUs. The script heavily relies on the Unsloth library, which provides custom kernels to accelerate training by approximately two times and reduce VRAM usage by up to70% compared to standard Hugging Face implementations.

The Low-Rank Adaptation (LoRA) is configured with a rank of 16, which determines the capacity of the adapter matrix—a balanced choice for maintaining performance while keeping the adapter size small. Adapters are strategically injected into the model's core layers, including the attention mechanisms (q_proj, k_proj, v_proj, o_proj) and the feed-forward network layers (gate_proj, up_proj, down_proj).

For the training data, a 10% subset of the yahma/alpaca-cleaned instruction-following dataset is used, which helps in generalizing the model's conversational and instruction-adherence abilities. All training data is strictly formatted according to the Alpaca Prompt template (Instruction/Input/Response). Furthermore, the maximum context length is set to a generous2048 tokens (max_seq_length), with Unsloth automatically applying advanced RoPE scaling (xPos) to effectively handle this extended context without positional instability. The training is set to run for just one epoch (num_train_epochs = 1) for rapid demonstration and initial fine-tuning.

The Weight and Biases is used for the Model Metric Visualisation.
