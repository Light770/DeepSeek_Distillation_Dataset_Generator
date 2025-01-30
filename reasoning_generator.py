import json
import logging
import os
import time
from typing import List, Dict, Tuple

import numpy as np
import replicate
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-r1"
DEFAULT_NUM_EXAMPLES = 50
DEFAULT_OUTPUT_FILE = "reasoning_training_data.json"
DEFAULT_DELAY = 1.0
MAX_RETRIES = 3

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
    # Class variables with type hints
    model: str
    response_schemas: List[ResponseSchema]
    output_parser: StructuredOutputParser

    def __init__(self, api_key: str) -> None:
        """
        Initialize the reasoning dataset generator.
        
        Args:
            api_key (str): Replicate API token
            
        Raises:
            ValueError: If API token is invalid
        """
        # API key is handled by environment variable
        self.model = DEFAULT_MODEL_NAME
        
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
        
        template = """You are an expert reasoning assistant. Generate a training example in valid JSON format.

{format_guidelines}

1. {domain_prompt}
2. Provide a detailed step-by-step reasoning process
3. State the final answer clearly
4. Assign a confidence score (0-10) based on the reasoning reliability
5. Include a verification step to check the reasoning

Your response MUST be valid JSON wrapped in ```json code blocks:

```json
{
    "problem": "The problem statement",
    "step_by_step": "1. First step\\n2. Second step\\n3. Third step",
    "final_answer": "The final answer",
    "confidence": 8,
    "verification": "Verification of the steps and answer"
}
```

Generate a single, high-quality example with careful reasoning."""
        
        return PromptTemplate(
            template=template,
            input_variables=[],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions(),
                "format_guidelines": format_guidelines,
                "domain_prompt": domain_prompts.get(domain, domain_prompts["general"])
            }
        )
    
    def generate_examples(self, num_examples: int, output_file: str, domain: str = "general", 
                         delay: float = DEFAULT_DELAY, max_retries: int = MAX_RETRIES) -> Tuple[List[Dict], Dict]:
        """Generate multiple reasoning examples and save them to a file."""
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
        
        logger.info(f"Generating {num_examples} reasoning examples...")
        
        for _ in tqdm(range(num_examples)):
            retries = 0
            while retries < max_retries:
                response_text = ""  # Initialize response_text here
                try:
                    # Generate example
                    system_prompt = "You are an expert reasoning assistant that breaks down problems step by step."
                    full_prompt = prompt.format()  # Remove the extra formatting
            
                    # Collect the streamed response
                    response_text = ""
                    for event in replicate.stream(
                        self.model,
                        input={
                            "prompt": f"{system_prompt}\n\n{full_prompt}",
                            "temperature": 0.7,
                            "max_tokens": 2000,  # Increased token limit
                            "stop": None  # Allow complete response
                        }
                    ):
                        response_text += str(event)

                    # Clean up the response text
                    response_text = response_text.strip()
                    logger.debug(f"Raw response:\n{response_text}")

                    # Extract JSON if wrapped in code blocks
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_text = response_text.split("```")[1].split("```")[0].strip()
                    else:
                        json_text = response_text

                    try:
                        # First try direct JSON parsing
                        parsed_response = json.loads(json_text)
                        
                        # Validate required fields
                        required_fields = ['problem', 'step_by_step', 'final_answer', 'confidence', 'verification']
                        missing_fields = [field for field in required_fields if field not in parsed_response]
                        
                        if missing_fields:
                            raise ValueError(f"Missing required fields: {missing_fields}")
                    except Exception as parse_error:
                        logger.error(f"Parsing error: {str(parse_error)}")
                        logger.debug(f"Failed to parse:\n{response_text}")
                        # Add basic response cleaning
                        if "```json" in response_text:
                            # Extract JSON if it's in a code block
                            json_text = response_text.split("```json")[1].split("```")[0]
                            try:
                                parsed_response = json.loads(json_text)
                            except json.JSONDecodeError:
                                raise ValueError("Failed to parse JSON from code block")
                        else:
                            raise ValueError("Invalid response format")
                    # Validate the generated example
                    format_valid, format_msg = validator.validate_format(parsed_response)
                    steps_valid, steps_msg = validator.validate_reasoning_steps(parsed_response)
            
                    if format_valid and steps_valid:
                        examples.append(parsed_response)
                        generation_stats['successes'] += 1
                        break  # Exit retry loop on success
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
                            logger.warning(f"Validation failed, retrying ({retries}/{max_retries}): {format_msg} {steps_msg}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error generating example: {str(e)}")
                    if response_text:  # Only log response_text if it exists
                        logger.debug(f"Failed response:\n{response_text}")
                    generation_stats['api_errors'] += 1
                    retries += 1
                    if retries < max_retries:
                        logger.info(f"Retrying ({retries}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    break
            
                # Save after each successful generation
                generation_stats['attempts'] += 1
                with open(output_file, 'w') as f:
                    json.dump(examples, f, indent=2)
            
                # Rate limiting
                time.sleep(delay)
        
        # Calculate final quality metrics
        quality_metrics = validator.calculate_quality_metrics(examples)
        generation_stats.update(quality_metrics)
                
        return examples, generation_stats

def main():
    try:
        # Load API token from environment variable
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("Please set the REPLICATE_API_TOKEN environment variable")
        
        # Initialize generator
        generator = ReasoningDatasetGenerator(api_token)
        
        # Generate dataset
        logger.info(f"Generating {DEFAULT_NUM_EXAMPLES} reasoning examples...")
        examples, stats = generator.generate_examples(
            num_examples=DEFAULT_NUM_EXAMPLES,
            output_file=DEFAULT_OUTPUT_FILE
        )
        
        print(f"Dataset generated and saved to {DEFAULT_OUTPUT_FILE}")
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
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
