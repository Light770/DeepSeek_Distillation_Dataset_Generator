from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
import os
from typing import List, Dict
import time
from tqdm import tqdm

class RLDatasetGenerator:
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        """
        Initialize the RL dataset generator.
        
        Args:
            api_key (str): Anthropic API key
            model_name (str): Model name to use (default: claude-3-haiku-20240307)
        """
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model_name,
            max_tokens=1024
        )
        
        # Define the output schema for structured responses
        self.response_schemas = [
            ResponseSchema(name="input", description="The input scenario or question"),
            ResponseSchema(name="response", description="The model's response"),
            ResponseSchema(name="reward", description="A numerical reward score (0-10)"),
            ResponseSchema(name="feedback", description="Feedback explaining the reward score")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for generating scenarios and responses."""
        template = """Generate a training example for reinforcement learning with the following components:

1. Create a realistic input scenario or question
2. Provide a response to that input
3. Assign a reward score (0-10) based on the quality of the response
4. Provide feedback explaining the reward score

{format_instructions}

Generate a single, high-quality example that would be useful for training an AI assistant."""
        
        return PromptTemplate(
            template=template,
            input_variables=[],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def generate_examples(self, num_examples: int, output_file: str, delay: float = 0.5) -> List[Dict]:
        """
        Generate multiple examples and save them to a file.
        
        Args:
            num_examples (int): Number of examples to generate
            output_file (str): Path to save the dataset
            delay (float): Delay between API calls in seconds
            
        Returns:
            List[Dict]: Generated examples
        """
        prompt = self._create_prompt_template()
        examples = []
        
        for _ in tqdm(range(num_examples)):
            try:
                # Generate example
                response = self.llm.invoke(prompt.format())
                parsed_response = self.output_parser.parse(response.content)
                examples.append(parsed_response)
                
                # Save after each successful generation
                with open(output_file, 'w') as f:
                    json.dump(examples, f, indent=2)
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error generating example: {str(e)}")
                continue
                
        return examples

def main():
    # Load API key from environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
    
    # Initialize generator
    generator = RLDatasetGenerator(api_key)
    
    # Generate dataset
    num_examples = 100  # Adjust as needed
    output_file = "rl_training_data.json"
    
    print(f"Generating {num_examples} examples...")
    examples = generator.generate_examples(num_examples, output_file)
    
    print(f"Dataset generated and saved to {output_file}")
    print(f"Total examples generated: {len(examples)}")

if __name__ == "__main__":
    main()
