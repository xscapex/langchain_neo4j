from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()


# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_KEY = os.getenv("NEO4J_KEY")
NEO4J_URL = os.getenv("NEO4J_URL")

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

@cl.langchain_factory
def factory():
    
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username="neo4j",
        password=NEO4J_KEY
    )

    llm_chain =GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True
)

    return llm_chain