import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os
from typing import Dict, List
import numpy as np

class Llama3Trainer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-13b-hf",  # placeholder for Llama 3 model
        output_dir: str = "./llama3-ft-output",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        # Updated LoRA config for Llama 3
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Extended target modules
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=50,
            logging_steps=10,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            eval_steps=50,
            warmup_ratio=0.03,
            weight_decay=0.1,
            fp16=True,
            bf16=True,  # Added mixed precision training
            optim="adamw_torch_fused",  # Using fused AdamW for better performance
        )
        
    def prepare_model(self):
        """Initialize and prepare the model for training"""
        print("Loading Llama 3 base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Using bfloat16 for better numerical stability
            device_map="auto",
            load_in_8bit=True,
            attn_implementation="flash_attention_2"  # Using Flash Attention 2 for better performance
        )
        
        print("Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        print("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, self.lora_config)
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_and_process_data(self, json_path: str):
        """Load and process the RL training data"""
        print("Loading training data...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        formatted_data = []
        for example in data:
            prompt = f"""# Input Scenario
{example['input_scenario']}

# Expected Response Quality
Target reward score: {example['reward']}/10

# Previous Feedback
{example['feedback']}"""
            
            completion = example['response']
            
            formatted_data.append({
                'text': f"{prompt}\n\nResponse: {completion}{self.tokenizer.eos_token}"
            })
            
        self.dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=4096,  # Increased context window for Llama 3
                padding="max_length"
            )
            
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            remove_columns=self.dataset.column_names,
            batch_size=8,
            num_proc=4
        )
        
    def train(self):
        """Train the model"""
        print("Starting training...")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        trainer.train()
        
        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        
def main():
    trainer = Llama3Trainer(
        output_dir="./llama3-ft-output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8
    )
    
    trainer.prepare_model()
    trainer.load_and_process_data("rl_training_data.json")
    
    trainer.train()

if __name__ == "__main__":
    main()
