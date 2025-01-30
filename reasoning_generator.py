import replicate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
import os
from typing import List, Dict
import time
from tqdm import tqdm

class ReasoningDatasetGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the reasoning dataset generator.
        
        Args:
            api_key (str): Replicate API token
        """
        # API key is handled by environment variable
        self.model = "deepseek-ai/deepseek-r1"
        
        # Define the output schema for structured responses
        self.response_schemas = [
            ResponseSchema(name="problem", description="The reasoning problem or question"),
            ResponseSchema(name="step_by_step", description="Step-by-step reasoning process"),
            ResponseSchema(name="final_answer", description="The final answer after reasoning"),
            ResponseSchema(name="confidence", description="A confidence score (0-10) in the reasoning"),
            ResponseSchema(name="verification", description="Verification of the reasoning steps")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for generating reasoning examples."""
        template = """Generate a training example for a reasoning model with the following components:

1. Create a challenging problem that requires careful reasoning (math, logic, or analysis)
2. Provide a detailed step-by-step reasoning process
3. State the final answer clearly
4. Assign a confidence score (0-10) based on the reasoning reliability
5. Include a verification step to check the reasoning

{format_instructions}

Generate a single, high-quality example that demonstrates careful reasoning and problem-solving."""
        
        return PromptTemplate(
            template=template,
            input_variables=[],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def generate_examples(self, num_examples: int, output_file: str, delay: float = 1.0) -> List[Dict]:
        """
        Generate multiple reasoning examples and save them to a file.
        
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
                system_prompt = "You are an expert reasoning assistant that breaks down problems step by step."
                full_prompt = f"{system_prompt}\n\n{prompt.format()}"
                
                # Collect the streamed response
                response_text = ""
                for event in replicate.stream(
                    self.model,
                    input={"prompt": full_prompt}
                ):
                    response_text += str(event)
                
                parsed_response = self.output_parser.parse(response_text)
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
    # Load API token from environment variable
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("Please set the REPLICATE_API_TOKEN environment variable")
    
    # Initialize generator
    generator = ReasoningDatasetGenerator(api_key)
    
    # Generate dataset
    num_examples = 50  # Adjust as needed
    output_file = "reasoning_training_data.json"
    
    print(f"Generating {num_examples} reasoning examples...")
    examples = generator.generate_examples(num_examples, output_file)
    
    print(f"Dataset generated and saved to {output_file}")
    print(f"Total examples generated: {len(examples)}")

if __name__ == "__main__":
    main()
