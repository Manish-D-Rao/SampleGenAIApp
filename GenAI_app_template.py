import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # Or ChatGroq / ChatOllama

# 1. Load environment variables
load_dotenv()

# 2. Initialize your LLM (Choose your provider)
llm = ChatOpenAI(model="gpt-4.1-nano")  # or ChatGroq, ChatOllama, etc.

# 3. Define the Prompt Template using Tuple Shorthand (The Modern Standard)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful expert AI assistant. Answer clearly.",
        ),
        ("user", "Question: {question}"),
    ]
)

# 4. Define the Output Parser
output_parser = StrOutputParser()

# 5. Build the LCEL Chain (Pipe operator '|')
chain = prompt | llm | output_parser

# 6. Invoke the chain with your input dictionary
response = chain.invoke({"question": "Explain how Transformers work in 1 sentence."})
print(response)
