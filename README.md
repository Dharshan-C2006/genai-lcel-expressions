# Design and Implementation of LangChain Expression Language (LCEL) Expressions

## AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

## PROBLEM STATEMENT:
Traditional language model piping structures require verbose nested functions to parse outputs from prompts to an LLM. This implementation replaces that paradigm with a clean, declarative pipeline using **LangChain Expression Language (LCEL)**. The pipeline takes two distinct user inputs (`ingredient` and `meal_type`) and securely chains three core modular components—`ChatPromptTemplate`, `ChatOpenAI`, and `StrOutputParser`—to systematically generate customized, structured recipes across single invoke and multi-request batch instances.

## DESIGN STEPS:

### STEP 1:
**Environment Setup & Configuration:** Configure system environment bindings using `dotenv` variables to securely manage API keys. Mask credentials away from public source files to satisfy secure repository guidelines.

### STEP 2:
**Component Architecture Definition:** Instanciate the separate phases of our LangChain workflow engine:
* **Prompt Definition:** Create a `ChatPromptTemplate` using exactly two target criteria keys: `{ingredient}` and `{meal_type}`.
* **Inference Engine Initialization:** Instantiate a `ChatOpenAI` chat completions instance targeting a specific fast-response model snapshot.
* **Response Processing Phase:** Instantiate a clean structural sequence via `StrOutputParser` to isolate content text blocks from payload metadata headers.

### STEP 3:
**Workflow Pipeline Graph Composition:** Form the cohesive operational chain using explicit pipe syntax (`|`). Run performance evaluation loops across single processing items (`.invoke()`) and multiple elements concurrently (`.batch()`) to verify complete multi-parameter stability.

## PROGRAM:

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Secure Environment Key Loading
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. Component Design (Utilizing exactly two prompt parameters: ingredient and meal_type)
prompt = ChatPromptTemplate.from_template(
    "Create a quick and unique {meal_type} recipe centering around the ingredient: {ingredient}. "
    "Include a catchy name and a 2-sentence summary."
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
output_parser = StrOutputParser()

# 3. Structural LCEL Pipe Compilation
lcel_chain = prompt | model | output_parser

# 4. Multi-Interface Evaluation Real-World Scenarios
if __name__ == "__main__":
    # Standard Invocation Scenario
    print("--- Single Invocation Execution ---")
    print(lcel_chain.invoke({"ingredient": "avocado", "meal_type": "breakfast"}))
    
    # Concurrent Batch Request Processing
    print("\n--- Parallel Batch Processing Execution ---")
    batch_scenarios = [
        {"ingredient": "dark chocolate", "meal_type": "dessert"},
        {"ingredient": "chickpeas", "meal_type": "dinner"}
    ]
    batch_outputs = lcel_chain.batch(batch_scenarios)
    for index, response in enumerate(batch_outputs, 1):
        print(f"Recipe Variant {index}:\n{response}\n")
```

## OUTPUT:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/055bbf68-b78c-4659-84e4-b5f88a7b5a17" />

## RESULT:
Thus, a modernized declarative LangChain Expression Language (LCEL) application pipeline successfully accepted multiple custom prompt constraints to dynamically generate contextually accurate recipes, satisfying all code validation constraints with masked API keys.
