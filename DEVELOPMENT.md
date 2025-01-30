# Development Guide for Open-R1

This guide provides detailed steps for setting up your development environment and contributing to the Open-R1 project.

## Prerequisites

1. Python Environment
   ```bash
   # Install Python 3.8 or higher
   python --version  # Verify Python version
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Required Accounts
   - GitHub account
   - Replicate API account (for model access)
   - Hugging Face account (optional, for model hosting)

## Initial Setup

1. Clone the Repository
   ```bash
   git clone https://github.com/your-org/open-r1.git
   cd open-r1
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set Up Environment Variables
   ```bash
   # Create .env file
   echo "REPLICATE_API_TOKEN=your_token_here" > .env
   
   # Load environment variables
   source .env  # On Windows: set /p REPLICATE_API_TOKEN=<.env
   ```

## Development Workflow

1. Create a New Branch
   ```bash
   # Create and switch to a new branch
   git checkout -b feature/your-feature-name
   ```

2. Code Style Guidelines
   - Use Python type hints
   - Follow PEP 8 style guide
   - Maximum line length: 80 characters
   - Use descriptive variable names
   - Add docstrings for functions and classes

3. Running Tests
   ```bash
   # Install test dependencies
   pip install pytest pytest-cov

   # Run tests with coverage
   pytest --cov=.
   ```

4. Local Development
   ```bash
   # Run linting
   flake8 .
   
   # Format code
   black .
   
   # Type checking
   mypy .
   ```

## Contributing Code

1. Making Changes
   - Write clear, documented code
   - Add appropriate tests
   - Update documentation as needed
   - Keep commits focused and atomic

2. Commit Messages
   ```bash
   # Format: type(scope): description
   git commit -m "feat(generator): add support for math problems"
   git commit -m "fix(validation): correct confidence score parsing"
   ```

3. Push Changes
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create Pull Request
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out PR template
   - Request review

## Project Structure

```
open-r1/
├── docs/
│   ├── CONTRIBUTING.md
│   ├── DEVELOPMENT.md
│   └── mechanistic_interpretability.md
├── src/
│   ├── generator/
│   │   └── reasoning_generator.py
│   ├── validation/
│   │   └── validator.py
│   └── utils/
│       └── helpers.py
├── tests/
│   └── test_generator.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Common Tasks

1. Generate Reasoning Dataset
   ```python
   from src.generator.reasoning_generator import ReasoningDatasetGenerator
   
   generator = ReasoningDatasetGenerator()
   examples, stats = generator.generate_examples(
       num_examples=50,
       output_file="data/reasoning_examples.json"
   )
   ```

2. Validate Dataset
   ```python
   from src.validation.validator import ReasoningDatasetValidator
   
   validator = ReasoningDatasetValidator()
   metrics = validator.calculate_quality_metrics(examples)
   ```

## Troubleshooting

1. API Issues
   - Verify API token is set correctly
   - Check API rate limits
   - Ensure proper error handling

2. Common Errors
   - ImportError: Check virtual environment activation
   - ModuleNotFoundError: Verify requirements installation
   - ValueError: Check input validation

## Getting Help

1. Documentation
   - Read the project documentation
   - Check existing issues
   - Review pull request history

2. Community Support
   - Join Discord server
   - Post on GitHub Discussions
   - Ask in Hugging Face Forums

## Best Practices

1. Code Quality
   - Write self-documenting code
   - Add comprehensive tests
   - Use type hints consistently
   - Follow project conventions

2. Documentation
   - Update README.md when needed
   - Document new features
   - Add examples for complex functionality
   - Keep API documentation current

3. Collaboration
   - Communicate changes early
   - Respond to review comments
   - Help review others' PRs
   - Share knowledge with team

## Release Process

1. Version Bumping
   ```bash
   # Update version in setup.py
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. Release Notes
   - Document breaking changes
   - List new features
   - Mention bug fixes
   - Credit contributors

## Next Steps

1. Pick a Task
   - Check open issues
   - Review project roadmap
   - Propose new features

2. Get Involved
   - Join community discussions
   - Attend team meetings
   - Share your progress

Remember to always check the latest documentation and follow project guidelines when contributing.
