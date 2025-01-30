import replicate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
import os
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
import numpy as np

class ReasoningDatasetValidator:
    """Validates reasoning examples and calculates quality metrics."""
    
    def validate_reasoning_steps(self, example: Dict) -> Tuple[bool, str]:
        """
        Validate coherence of reasoning steps.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not example.get('step_by_step'):
            return False, "Missing reasoning steps"
            
        steps = example['step_by_step'].split('\n')
        if len(steps) < 2:
            return False, "Too few reasoning steps"
            
        # More robust step number detection
        for i, step in enumerate(steps, 1):
            step = step.strip()
            if not step:
                continue
            # Check for various numbering formats: "1.", "1)", "Step 1:", etc.
            valid_prefixes = [
                f"{i}.", f"{i})", f"Step {i}:", f"({i})"
            ]
            if not any(step.startswith(prefix) for prefix in valid_prefixes):
                return False, f"Step {i} not properly numbered. Should start with one of: {valid_prefixes}"
                
        return True, ""
    
    def validate_format(self, example: Dict) -> Tuple[bool, str]:
        """
        Verify output format compliance.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        required_fields = ['problem', 'step_by_step', 'final_answer', 
                         'confidence', 'verification']
        
        for field in required_fields:
            if field not in example:
                return False, f"Missing required field: {field}"
                
        try:
            confidence_str = str(example['confidence']).strip()
            # Handle potential string responses like "8 out of 10"
            confidence = float(confidence_str.split()[0])
            if not 0 <= confidence <= 10:
                return False, f"Confidence score {confidence} out of range (0-10)"
        except (ValueError, IndexError):
            return False, "Invalid confidence score format"
            
        return True, ""
    
    def calculate_quality_metrics(self, examples: List[Dict]) -> Dict:
        """
        Calculate quality metrics for dataset.
        
        Returns:
            Dict: Quality metrics
        """
        metrics = {
            'total_examples': len(examples),
            'avg_confidence': np.mean([float(ex['confidence']) for ex in examples]),
            'avg_steps': np.mean([len(ex['step_by_step'].split('\n')) for ex in examples]),
            'format_valid': 0,
            'steps_valid': 0
        }
        
        for ex in examples:
            format_valid, _ = self.validate_format(ex)
            steps_valid, _ = self.validate_reasoning_steps(ex)
            
            metrics['format_valid'] += int(format_valid)
            metrics['steps_valid'] += int(steps_valid)
            
        metrics['format_valid_pct'] = metrics['format_valid'] / len(examples) * 100
        metrics['steps_valid_pct'] = metrics['steps_valid'] / len(examples) * 100
        
        return metrics

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
        
    def _create_prompt_template(self, domain: str = "general") -> PromptTemplate:
        """
        Create the prompt template for generating reasoning examples.
        
        Args:
            domain (str): Reasoning domain ("math", "logic", "analysis", or "general")
        """
        domain_prompts = {
            "math": """Generate a mathematical reasoning problem that requires:
- Clear numerical or algebraic manipulation
- Multiple solution steps
- Application of mathematical concepts
- Careful attention to order of operations""",
            
            "logic": """Generate a logical reasoning problem that involves:
- Deductive or inductive reasoning
- If-then relationships
- Logical operators or set theory
- Clear premises and conclusions""",
            
            "analysis": """Generate an analytical problem that requires:
- Breaking down complex information
- Identifying patterns or relationships
- Drawing evidence-based conclusions
- Systematic evaluation of options""",
            
            "general": """Generate a challenging problem that requires careful reasoning (math, logic, or analysis)"""
        }
        
        format_guidelines = """
        Important formatting requirements:
        1. Number each reasoning step clearly (e.g., "1.", "2.", etc.)
        2. Provide confidence score as a single number between 0-10
        3. Ensure each step follows from the previous one
        4. Include clear verification at the end
        """
        
        template = f"""Generate a training example for a reasoning model with the following components:
        {format_guidelines}

1. {domain_prompts.get(domain, domain_prompts["general"])}
2. Provide a detailed step-by-step reasoning process
3. State the final answer clearly
4. Assign a confidence score (0-10) based on the reasoning reliability
5. Include a verification step to check the reasoning

{{format_instructions}}

Generate a single, high-quality example that demonstrates careful reasoning and problem-solving."""
        
        return PromptTemplate(
            template=template,
            input_variables=[],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def generate_examples(self, num_examples: int, output_file: str, domain: str = "general", 
                         delay: float = 1.0, max_retries: int = 3) -> Tuple[List[Dict], Dict]:
        """
        Generate multiple reasoning examples and save them to a file.
        
        Args:
            num_examples (int): Number of examples to generate
            output_file (str): Path to save the dataset
            delay (float): Delay between API calls in seconds
            
        Returns:
            List[Dict]: Generated examples
        """
        prompt = self._create_prompt_template(domain)
        examples = []
        validator = ReasoningDatasetValidator()
        
        generation_stats = {
            'attempts': 0,
            'successes': 0,
            'validation_failures': 0,
            'api_errors': 0,
            'validation_details': []
        }
        
        for _ in tqdm(range(num_examples)):
            retries = 0
            while retries < max_retries:
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
                # Validate the generated example
                format_valid, format_msg = validator.validate_format(parsed_response)
                steps_valid, steps_msg = validator.validate_reasoning_steps(parsed_response)
                
                if format_valid and steps_valid:
                    examples.append(parsed_response)
                    generation_stats['successes'] += 1
                else:
                    generation_stats['validation_failures'] += 1
                    generation_stats['validation_details'].append({
                        'attempt': generation_stats['attempts'],
                        'format_error': format_msg if not format_valid else None,
                        'steps_error': steps_msg if not steps_valid else None,
                        'raw_response': response_text[:200]  # First 200 chars for debugging
                    })
                    retries += 1
                    if retries < max_retries:
                        print(f"Validation failed, retrying ({retries}/{max_retries}): {format_msg} {steps_msg}")
                        continue
                    break
                
                # Save after each successful generation
                generation_stats['attempts'] += 1
                with open(output_file, 'w') as f:
                    json.dump(examples, f, indent=2)
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error generating example: {str(e)}")
                generation_stats['api_errors'] += 1
                continue
        
        # Calculate final quality metrics
        quality_metrics = validator.calculate_quality_metrics(examples)
        generation_stats.update(quality_metrics)
                
        return examples, generation_stats

def main():
    # Load API token from environment variable
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("Please set the REPLICATE_API_TOKEN environment variable")
    
    # Initialize generator
    generator = ReasoningDatasetGenerator(api_token)
    
    # Generate dataset
    num_examples = 50  # Adjust as needed
    output_file = "reasoning_training_data.json"
    
    print(f"Generating {num_examples} reasoning examples...")
    examples, stats = generator.generate_examples(num_examples, output_file)
    
    print(f"Dataset generated and saved to {output_file}")
    print(f"Generation Statistics:")
    print(f"- Total attempts: {stats['attempts']}")
    print(f"- Successful generations: {stats['successes']}")
    print(f"- Validation failures: {stats['validation_failures']}")
    print(f"- API errors: {stats['api_errors']}")
    print(f"\nQuality Metrics:")
    print(f"- Format valid: {stats['format_valid_pct']:.1f}%")
    print(f"- Steps valid: {stats['steps_valid_pct']:.1f}%")
    print(f"- Average confidence: {stats['avg_confidence']:.1f}")
    print(f"- Average steps: {stats['avg_steps']:.1f}")

if __name__ == "__main__":
    main()
