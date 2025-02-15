import json
import logging
import os
import time
import uuid
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

import numpy as np
import replicate
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, DOMAIN_PROMPTS

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

class APIClient:
    """API client for different model providers."""
    
    def __init__(self, api_key: str, provider: str = "replicate"):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif provider == "replicate":
            # Replicate uses environment variable
            os.environ["REPLICATE_API_TOKEN"] = api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate text using the selected API provider."""
        if self.provider == "deepseek":
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
            
        elif self.provider == "replicate":
            response = replicate.run(
                DEFAULT_MODEL_NAME,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9
                }
            )
            return "".join(str(chunk) for chunk in response if chunk)

class ReasoningDatasetGenerator:
    """Generates reasoning examples using configurable API providers."""
    
    def __init__(self, api_key: str, provider: str = "replicate", model: str = "deepseek-ai/deepseek-r1", domain: str = "general") -> None:
        """
        Initialize the reasoning dataset generator.
        
        Args:
            api_key (str): Replicate API token
            
        Raises:
            ValueError: If API token is invalid
        """
        # API key is handled by environment variable
        self.model = DEFAULT_MODEL_NAME
        
        # Initialize model and domain
        self.model = model
        self.domain = domain
        
    def _generate_prompt(self, domain: str = "general") -> str:
        """Generate the complete prompt for the model."""
        domain_prompt = DOMAIN_PROMPTS.get(self.domain, DOMAIN_PROMPTS["general"])
        return f"{SYSTEM_PROMPT}\n\n{domain_prompt}"

    def _parse_response(self, response_text: str, messages: List[Dict]) -> Dict:
        """Parse the model's response into the required schema format."""
        try:
            # Clean up the response text
            response_text = response_text.strip()
            
            # Find JSON content
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in response")
                
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            
            # Create new schema-compliant structure
            example = {
                "problem": parsed["problem"],
                "solution": parsed["steps"],
                "answer": parsed["answer"],
                "problem_type": "Algebra",  # Default for math problems
                "question_type": "math-word-problem",
                "source": "open-r1",
                "uuid": str(uuid.uuid4()),
                "is_reasoning_complete": [True],  # Single generation
                "generations": [response_text],
                "correctness_math_verify": [True],  # Would need actual verification
                "correctness_llama": None,  # Would need Llama verification
                "finish_reasons": ["stop"],
                "correctness_count": 1,
                "messages": messages
            }
            
            return example
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.debug(f"Response text:\n{response_text}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        """Create the prompt template for generating reasoning examples."""
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
{{
    "problem": "The problem statement",
    "step_by_step": "1. First step\\n2. Second step\\n3. Third step",
    "final_answer": "The final answer",
    "confidence": 8,
    "verification": "Verification of the steps and answer"
}}
```

Generate a single, high-quality example with careful reasoning."""

        domain_prompt = DOMAIN_PROMPTS.get(self.domain, DOMAIN_PROMPTS["general"])
        return f"{template}\n\n{domain_prompt}\n\n{format_guidelines}"
    
    def _extract_and_parse_json(self, response_text: str) -> dict:
        """Extract and parse JSON with detailed error handling."""
        if not response_text:
            raise ValueError("Empty response text")
            
        logger.debug(f"Attempting to parse response of length {len(response_text)}")
        
        json_text = None
        errors = []
        
        # Try multiple extraction methods
        try:
            # Method 1: ```json blocks
            if "```json" in response_text:
                parts = response_text.split("```json")
                if len(parts) > 1:
                    json_text = parts[1].split("```")[0].strip()
                    return json.loads(json_text)
        except json.JSONDecodeError as e:
            errors.append(f"Method 1 failed: {str(e)}")
        
        try:
            # Method 2: ``` blocks
            if "```" in response_text:
                parts = response_text.split("```")
                if len(parts) > 1:
                    json_text = parts[1].strip()
                    return json.loads(json_text)
        except json.JSONDecodeError as e:
            errors.append(f"Method 2 failed: {str(e)}")
        
        try:
            # Method 3: Find JSON structure
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                return json.loads(json_text)
        except json.JSONDecodeError as e:
            errors.append(f"Method 3 failed: {str(e)}")
        
        # If all methods failed, raise error with details
        error_msg = "\n".join(errors)
        logger.error(f"All JSON parsing methods failed:\n{error_msg}")
        raise ValueError(f"Could not parse JSON response. Attempts failed with: {error_msg}")

    def generate_examples(self,
                         num_examples: int,
                         output_file: str,
                         domain: str = "math",
                         temperature: float = 0.7,
                         max_retries: int = 3,
                         delay: float = 1.0) -> Tuple[List[Dict], Dict]:
        """Generate examples with updated schema."""
        examples = []
        stats = {
            'attempts': 0,
            'successes': 0,
            'validation_failures': 0,
            'api_errors': 0
        }
        
        logger.info(f"Generating {num_examples} examples...")
        
        for i in tqdm(range(num_examples)):
            messages = [
                {
                    "from": "user",
                    "value": self._generate_prompt(domain)
                }
            ]
            
            for attempt in range(max_retries):
                try:
                    # Initialize API client if not already done
                    if not hasattr(self, 'api_client'):
                        self.api_client = APIClient(self.api_key, self.provider)
                    
                    # Generate using selected API
                    response_text = await self.api_client.generate(
                        prompt=messages[0]["value"],
                        temperature=temperature,
                        max_tokens=1000
                    )
                    
                    # Add assistant response to messages
                    messages.append({
                        "from": "assistant",
                        "value": response_text
                    })
                    
                    # Parse and validate with new schema
                    example = self._parse_response(response_text, messages)
                    examples.append(example)
                    stats['successes'] += 1
                    
                    # Save progress
                    with open(output_file, 'w') as f:
                        json.dump(examples, f, indent=2)
                        
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    logger.error(f"Error generating example {i}, attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate example {i} after {max_retries} attempts")
                    else:
                        time.sleep(delay)
                        continue
            
            # Rate limiting between examples
            if i < num_examples - 1:
                time.sleep(delay)
        
        return examples
        """Generate multiple reasoning examples with robust error handling."""
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
            
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
        
        for example_num in tqdm(range(num_examples)):
            retries = 0
            success = False
            response_text = ""  # Initialize response_text here
            
            while retries < max_retries and not success:
                try:
                    # 1. Generate API request
                    system_prompt = "You are an expert reasoning assistant that breaks down problems step by step."
                    try:
                        full_prompt = prompt.format()  # No arguments needed now
                    except Exception as prompt_error:
                        logger.error(f"Error formatting prompt: {str(prompt_error)}")
                        raise ValueError(f"Prompt formatting failed: {str(prompt_error)}")
                    
                    if not full_prompt.strip():
                        raise ValueError("Empty prompt generated")
                        
                    logger.debug(f"Generated prompt:\n{full_prompt}")
                    
                    # 2. Make API call
                    try:
                        prediction = replicate.run(
                            self.model,
                            input={
                                "prompt": f"{system_prompt}\n\n{full_prompt}",
                                "temperature": 0.7,
                                "max_tokens": 2000,
                                "top_p": 0.9,
                                "top_k": 50
                            }
                        )
                    except Exception as api_error:
                        logger.error(f"API call failed: {str(api_error)}")
                        generation_stats['api_errors'] += 1
                        raise
                    
                    # Collect the complete response
                    response_text = ""
                    for item in prediction:
                        if item is not None:
                            response_text += str(item)

                    # Parse the response using our helper method
                    parsed_response = self._extract_and_parse_json(response_text)
                    
                    # Validate required fields
                    required_fields = ['problem', 'step_by_step', 'final_answer', 'confidence', 'verification']
                    missing_fields = [field for field in required_fields if field not in parsed_response]
                    
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
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
        # Check for API tokens
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        deepseek_token = os.getenv("DEEPSEEK_API_KEY")
        
        if not (replicate_token or deepseek_token):
            logger.error("Missing API tokens")
            raise ValueError("Please set either REPLICATE_API_TOKEN or DEEPSEEK_API_KEY environment variable")
        
        # Select provider based on available tokens
        if deepseek_token:
            provider = "deepseek"
            api_token = deepseek_token
        else:
            provider = "replicate"
            api_token = replicate_token
        
        # Initialize generator
        generator = ReasoningDatasetGenerator(api_token)
        
        # Verify output directory exists
        output_dir = os.path.dirname(DEFAULT_OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate dataset with progress tracking
        logger.info(f"Starting generation of {DEFAULT_NUM_EXAMPLES} examples...")
        examples, stats = generator.generate_examples(
            num_examples=DEFAULT_NUM_EXAMPLES,
            output_file=DEFAULT_OUTPUT_FILE
        )
        
        if not examples:
            logger.error("No examples were generated successfully")
            return
            
        logger.info(f"Generation complete. Stats: {stats}")
        
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
