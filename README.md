## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
LangChain Expression Language (LCEL) simplifies interactions with large language models (LLMs) by creating reusable and structured expressions. This task involves:

Designing an LCEL expression with dynamic prompt parameters (e.g., topic and length).
Using three essential components: Prompt- A structured input with placeholders for parameters, Model- An LLM used to process the prompt and Output Parser- A parser to interpret the model's output.
Demonstrating the LCEL expression's functionality in generating structured, relevant outputs.
### DESIGN STEPS:

STEP 1: Define the Parameters
Identify the parameters (topic and length) to allow dynamic customization of prompts.

STEP 2: Design the Prompt Template
Create a structured prompt template with placeholders for parameters.

STEP 3: Select the Model
Use an LLM, such as OpenAI's GPT, to process the prompt.

STEP 4: Implement the Output Parser
Design an output parser to format and structure the model's output.

STEP 5: Integrate Components into an LCEL Expression
Combine the prompt template, model, and output parser into a LangChain pipeline.

STEP 6: Evaluate with Examples
Test the LCEL expression using multiple input values for topic and length.

### PROGRAM:
```
!pip install -U langchain langchain-openai langchain-community docarray


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

output_parser = StrOutputParser()



prompt = ChatPromptTemplate.from_template(
    "Give three interesting facts about {topic}."
)

chain = prompt | model | output_parser

response = chain.invoke(
    {
        "topic": "Artificial Intelligence"
    }
)

print("========== Example 1 ==========")
print(response)



embeddings = OpenAIEmbeddings()

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "The Amazon rainforest is the largest tropical rainforest in the world.",
        "It covers about 5.5 million square kilometers across South America.",
        "The Amazon River is one of the longest rivers in the world."
    ],
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

template = """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    RunnableMap(
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"]
        }
    )
    | prompt
    | model
    | output_parser
)

response = retrieval_chain.invoke(
    {
        "question": "Which is the largest tropical rainforest in the world?\n"
    }
)

print("\n========== Example 2 ==========")
print(response)



response_schemas = [
    ResponseSchema(
        name="summary",
        description="A concise summary of the topic"
    ),
    ResponseSchema(
        name="word_count",
        description="Total number of words in the summary"
    )
]

structured_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)

format_instructions = structured_output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    input_variables=["topic", "length"],
    partial_variables={
        "format_instructions": format_instructions
    },
    template="""
You are a helpful assistant.

Write a concise summary about {topic} in approximately {length} words.

{format_instructions}
"""
)

chain = LLMChain(
    llm=model,
    prompt=prompt_template
)

response = chain.invoke(
    {
        "topic": "Artificial Intelligence",
        "length": "100"
    }
)

parsed_output = structured_output_parser.parse(
    response["text"]
)

print(parsed_output)


```

### OUTPUT:

<img width="1252" height="506" alt="Screenshot 2026-08-05 114028" src="https://github.com/user-attachments/assets/db716c67-33dd-4875-ba06-46379bb16c02" />



### RESULT:

Thus, the LangChain Expression Language (LCEL) expression that utilizes two prompt parameters and three key components (prompt, model, and output parser) was designed and implemented successfully. And also evaluated its functionality by analyzing relevant examples of its application in real-world scenarios.

