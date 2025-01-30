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

class LLaMATrainer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-13b-hf",  # or your local path
        output_dir: str = "./llama-ft-output",
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
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
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
        )
        
    def prepare_model(self):
        """Initialize and prepare the model for training"""
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
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
            
        # Format data for training
        formatted_data = []
        for example in data:
            # Format similar to previous script
            prompt = f"""# Input Scenario
{example['input_scenario']}

# Expected Response Quality
Target reward score: {example['reward']}/10

# Previous Feedback
{example['feedback']}"""
            
            completion = example['response']
            
            # Combine prompt and completion with special tokens
            formatted_data.append({
                'text': f"{prompt}\n\nResponse: {completion}{self.tokenizer.eos_token}"
            })
            
        # Create dataset
        self.dataset = Dataset.from_list(formatted_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=2048,
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
        
        # Save the final model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        
def main():
    # Initialize trainer
    trainer = LLaMATrainer(
        output_dir="./llama-ft-output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8
    )
    
    # Prepare model and load data
    trainer.prepare_model()
    trainer.load_and_process_data("rl_training_data.json")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()