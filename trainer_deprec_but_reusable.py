import json
import os
from typing import List, Dict, Union
import replicate
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class TrainingExample:
    input_scenario: str
    response: str
    reward: float
    feedback: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to format expected by LLaMA training"""
        prompt = f"""# Input Scenario
{self.input_scenario}

# Expected Response Quality
Target reward score: {self.reward}/10

# Previous Feedback
{self.feedback}"""

        completion = self.response

        return {
            "prompt": prompt,
            "completion": completion
        }

class DatasetProcessor:
    def __init__(self, output_path: str = "training_data.jsonl"):
        self.output_path = output_path
        self.examples: List[TrainingExample] = []

    def add_example(self, example: TrainingExample):
        """Add a single training example"""
        self.examples.append(example)

    def add_examples_from_csv(self, csv_path: str):
        """Load examples from a CSV file"""
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            example = TrainingExample(
                input_scenario=row['input_scenario'],
                response=row['response'],
                reward=float(row['reward']),
                feedback=row['feedback']
            )
            self.examples.append(example)

    def save_jsonl(self):
        """Save examples in JSONL format required by LLaMA"""
        with open(self.output_path, 'w') as f:
            for example in self.examples:
                json_line = json.dumps(example.to_dict())
                f.write(json_line + '\n')
        return self.output_path

def train_model(
    training_data_path: str,
    destination: str,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 1,
    warmup_ratio: float = 0.03,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1
):
    """Initialize and start training using replicate"""
    training = replicate.trainings.create(
        version="replicate/llama-13b-lora:455d66312a66299fba685548fe24f66880f093007b927abd19f4356295f8577c",
        input={
            "train_data": training_data_path,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_ratio": warmup_ratio,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": "q_proj,v_proj"
        },
        destination=destination
    )
    return training

def main():
#     # Initialize dataset processor
#     processor = DatasetProcessor()

#     # Add example (you can add more examples or load from CSV)
#     example = TrainingExample(
#         input_scenario="""A customer service representative receives the following message:
# "I've been trying to access my account for the past hour but keep getting an error message saying 'Invalid credentials'. 
# I've reset my password twice already and still can't get in. This is ridiculous! I need to submit an important application by tonight's deadline."""
#         response="""I understand how frustrating this must be, especially with an urgent deadline approaching. Let me help you resolve this right away.

# First, to clarify - are you getting this error message on our website or mobile app? Also, after resetting your password, 
# are you waiting at least 5 minutes before trying to log in again? Our system needs this time to fully process password changes.

# In the meantime, I can check if there are any known system issues affecting logins. Could you please provide your email 
# address associated with the account?

# Rest assured, if you're unable to submit your application due to these technical difficulties, I'll make a note in your 
# account about the login issues so the application team is aware of the circumstances.""",
#         reward=8.0,
#         feedback="""This response earned a high score (8/10) for:
# 1. Emotional Intelligence - Acknowledged frustration and urgency
# 2. Technical Problem-Solving - Asked clarifying questions and provided specific insights
# 3. Proactive Support - Offered system status check and documentation
# 4. Areas for Improvement - Could have provided alternative contact methods and escalation options"""
#     )
#     processor.add_example(example)

#     # Save to JSONL
#     training_data_path = processor.save_jsonl()

    # Start training
    training = train_model(
        training_data_path="rl_training_data.json",
        destination="light770/customer-service-assistant",  # Replace with your username
        train_batch_size=1,
        num_train_epochs=3  # Adjust based on your needs
    )
    
    print(f"Training started: {training}")

if __name__ == "__main__":
    main()
