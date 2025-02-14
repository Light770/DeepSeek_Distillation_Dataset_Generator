# Open-R1: Reasoning Dataset Generator

A tool for generating and validating high-quality reasoning examples using large language models.

## Overview

Reasoning Dataset Generator is a Python-based tool that generates structured reasoning examples across multiple domains including mathematics, logic, and analytical reasoning. It uses the Replicate API  or DeepSeek API to interact with language models and includes robust validation and quality control mechanisms.

## Features

- Generate reasoning examples with step-by-step solutions
- Multiple reasoning domains (math, logic, analysis)
- Validation and quality metrics
- JSON-formatted output
- Error handling and retry mechanisms
- Logging and progress tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/Light770/DeepSeek_Distillation_Dataset_Generator.git
cd open-r1

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Set up your Replicate API token:
   ```bash
   set REPLICATE_API_TOKEN=your_token_here  # On Windows
   export REPLICATE_API_TOKEN=your_token_here  # On Unix/MacOS

   set DEEPSEEK_API_KEY=your_token_here  # On Windows
   export DEEPSEEK_API_KEY=your_token_here  # On Unix/MacOS
   ```

2. Configure logging and output settings in the code if needed.

## Usage

```python
from reasoning_generator import ReasoningDatasetGenerator

# Initialize generator
generator = ReasoningDatasetGenerator(api_token)

# Generate examples
examples, stats = generator.generate_examples(
    num_examples=50,
    output_file="reasoning_data.json",
    domain="math"  # or "logic", "analysis", "general"
)
```

## Output Format

Generated examples are saved in JSON format:
```json
{
    "problem": "Problem statement",
    "step_by_step": "1. First step\n2. Second step\n3. Third step",
    "final_answer": "The final answer",
    "confidence": 8,
    "verification": "Verification explanation"
}
```

## Project Structure

```
open-r1/
├── docs/
│   ├── CONTRIBUTING.md
│   ├── DEVELOPMENT.md
│   └── mechanistic_interpretability.md
├── src/
│   ├── reasoning_generator.py
│   └── prompts.py
├── requirements.txt
└── README.md
```

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guidelines.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Documentation

- [Development Guide](docs/DEVELOPMENT.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Mechanistic Interpretability](docs/mechanistic_interpretability.md)

## Requirements

- Python 3.8+
- Replicate API access
- Dependencies listed in requirements.txt

## License

MIT License

Copyright (c) 2024 Light770

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- DeepSeek-R1 team for inspiration
- Replicate for API access
- Contributors and maintainers
