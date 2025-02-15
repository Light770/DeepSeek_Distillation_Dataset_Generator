"""Prompt templates for reasoning generation."""

SYSTEM_PROMPT = """You are an expert reasoning assistant. Your task is to generate a clear, step-by-step solution to a problem.
Follow these requirements exactly:

1. Provide a clear problem statement
2. Break down the solution into numbered steps
3. Give a final answer
4. Rate your confidence (1-10)
5. Verify your solution

Format your response as a JSON object with these exact fields:
{
    "problem": "Problem statement here",
    "steps": ["1. First step", "2. Second step", "..."],
    "answer": "Final answer here",
    "confidence": 8,
    "verification": "Verification explanation here"
}"""

DOMAIN_PROMPTS = {
    "math": """Generate a mathematical problem that requires:
- Clear numerical or algebraic manipulation
- Multiple solution steps
- Mathematical concepts from one of: Algebra, Number Theory, Geometry, Calculus
- Can be either a word problem or MCQ format
- Similar to problems found in math olympiads or AoPS forums

Format your response to clearly separate:
1. The problem statement
2. Step-by-step solution
3. Final answer in a concise format""",
    
    "logic": """Generate a logical reasoning problem with:
- Deductive reasoning
- If-then relationships
- Logical operators
- Clear premises""",
    
    "analysis": """Generate an analytical problem requiring:
- Information breakdown
- Pattern identification
- Evidence-based conclusions
- Systematic evaluation""",
    
    "general": """Generate a challenging reasoning problem (math, logic, or analysis)"""
}
