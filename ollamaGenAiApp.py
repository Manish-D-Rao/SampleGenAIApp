import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You r a helpful lead ai engineer.Please answer all my doubts."),
        ("user", "Question:{question}"),
    ]
)

st.title("Langchain Demo with gemma:2b model")

input_text = st.text_input("What's The Question You Have In Your Mind?")

llm = OllamaLLM(model="gemma:2b")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
