import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Initialize Environment and LLM
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano")

# 2. Ingest External Data (Load Source Document)
loader = WebBaseLoader("https://jalammar.github.io/illustrated-transformer/")
docs = loader.load()

# 3. Split Text into Bite-Sized Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 4. Generate Embeddings & Build Vector Database Index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)

# 5. Convert Vector Store into a Search Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. Define Structured Prompt Template
system_prompt = (
    "You are an expert AI engineering assistant. Answer the user's question "
    "using *only* the provided context below. If you do not know the answer, "
    "state that you don't know.\n\n"
    "<context>\n{context}\n</context>"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 7. Build Chains (Document Stuffing Chain + Retrieval Chain)
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 8. Execute the Pipeline
response = rag_chain.invoke(
    {"input": "How does multi-head attention work in transformers?"}
)

print(response["answer"])
