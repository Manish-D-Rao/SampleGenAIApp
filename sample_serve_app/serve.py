from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="qwen/qwen3.8-27b", api_key=GROQ_API_KEY)

system_template = "Translate to {language}"
prompt = ChatPromptTemplate([("system", system_template), ("user", "{text}")])


parser = StrOutputParser()

chain = prompt | model | parser

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
