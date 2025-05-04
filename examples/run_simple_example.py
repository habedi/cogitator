import logging

from cogitator import SelfConsistency, OllamaLLM

# Step 1: Configure logging (optional, but helpful)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTPX logs

# Step 2: Initialize the LLM (using Ollama)
# Needs Ollama running locally with the model pulled (e.g., `ollama pull gemma3:4b`)
try:
    llm = OllamaLLM(model="gemma3:4b")
except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    print("Please make sure Ollama is running and the model is pulled.")
    exit(1)

# Step 3: Choose a CoT strategies (Self-Consistency in this case)
# Self-Consistency generates multiple reasoning paths and finds the most common answer
sc_strategy = SelfConsistency(
    llm,
    n_samples=5,  # Number of reasoning paths to generate
    temperature=0.7  # Higher temperature can lead to more diverse answers
)

# Step 4: Define the prompt (with a basic CoT trigger)
question = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
prompt = f"Q: {question}\nA: Let's think step by step."

# Step 5: Run the CoT prompting sc_strategy
print(f"\nQuestion: {question}")
print("Running Self-Consistency CoT...")
final_answer = sc_strategy.run(prompt)  # Returns the most consistent (repeated) answer

# Expected output: $0.05 or 0.05 (may vary slightly based on model and temperature)
print(f"\nCogitator's Answer (Self-Consistency): {final_answer}")
