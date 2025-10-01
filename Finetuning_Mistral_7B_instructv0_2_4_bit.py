

!pip install "unsloth[colab-new]" torch transformers trl peft accelerate bitsandbytes

# QLoRA Fine-Tuning Script for Mistral 7B on Free GPU on colab
# This script uses the unsloth library for fast and memory-efficient fine-tuning
# of the Mistral-7B model using 4-bit quantization (QLoRA).

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
import os

# Model Config
max_seq_length = 2048 # Max context length. Unsloth handles RoPE scaling automatically.
dtype = None
load_in_4bit = True   # Enable 4-bit quantization

#  Mistral 7B
model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
print(f"Loading model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA Config & LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
)

# Dataset Preparation
# Using a cleaned version of the Alpaca instruction dataset
dataset_name = "yahma/alpaca-cleaned"
print(f"Loading dataset: {dataset_name}...")
dataset = load_dataset(dataset_name, split = "train[0%:10%]") # Using 10% of the dataset

# Define the Alpaca prompt template to format the data
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # The eos token

def formatting_prompts_func(examples):
    """Formats the instruction/input/output columns into a single 'text' column for training."""
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        if input:
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        else:
            text = alpaca_prompt.format(instruction, "", output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }
dataset = dataset.map(formatting_prompts_func, batched = True,)

# sample of the formatted data
print("\n--- Example of Formatted Training Text ---")
print(dataset[0]["text"])
print("\n")

#Training Arguments & Setup
output_dir = "mistral_7b_lora_finetuned"

training_args = TrainingArguments(
    per_device_train_batch_size = 2,           # Reduced batch size for low memory
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 25,
    optim = "adamw_8bit",
    seed = 42,
    output_dir = output_dir,
)

# SFT-Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
)

# Training
print("--- Starting Fine-Tuning (QLoRA) ---")
trainer.train()

print(f"\n Training complete! Adapters saved to ./{output_dir}")

# Saving the Model (LoRA Adapters Only)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and Tokenizer adapters saved successfully to: {output_dir}")

# Inference

def generate_response(instruction, input_text=""):
    """Generates a response from the fine-tuned model based on an instruction."""
    FastLanguageModel.for_inference(model) # Prepare the model for faster inference

    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer(
    [
        prompt,
    ], return_tensors = "pt").to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n[Model Generating Response...]\n")
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens = 256, use_cache = True)
    print("\n------------------------------\n")

test_instruction = "Explain the concept of Low-Rank Adaptation (LoRA) in simple terms."
test_input = ""
print(f"--- Testing Fine-Tuned Model ---")
print(f"Prompt: {test_instruction}")

generate_response(test_instruction, test_input)

test_instruction = "Explain the transfer learning in ml."
test_input = ""

print(f"--- Testing Fine-Tuned Model ---")
print(f"Prompt: {test_instruction}")
generate_response(test_instruction, test_input)

test_instruction = "Explain the difference between transfer learning and finetuning in ml."
test_input = ""
print(f"--- Testing Fine-Tuned Model ---")
print(f"Prompt: {test_instruction}")

# Generate the response
generate_response(test_instruction, test_input)

# --- Optional: Save in 16-bit format (for merging later) ---
# If you want to merge the LoRA adapters into a full 16-bit model for deployment (e.g., using vLLM),
# use this step. NOTE: This requires more disk space and VRAM.
# output_merged = "mistral_7b_merged"
# if not os.path.exists(output_merged): os.makedirs(output_merged)
# print(f"\nSaving merged 16-bit model to ./{output_merged} (might take a few minutes)...")
# model.save_pretrained_merged(output_merged, tokenizer, max_seq_length = max_seq_length, save_method = "merged_16bit",)
# print("Merged 16-bit model saved.")