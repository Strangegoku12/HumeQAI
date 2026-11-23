import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
PDF_PATH = os.getenv("PDF_PATH")
BASE_DIR = os.getenv("BASE_DIR")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY is not set in the environment variables..")

if not GROQ_MODEL:
    raise Exception("GROQ_MODEL is not set in the environment variables..")

if not EMBEDDING_MODEL:
    raise Exception("EMBEDDING_MODEL is not set in the environment variables..")

if not CHROMA_PATH:
    raise Exception("CHROMA_PATH is not set in the environment variables..")

if not PDF_PATH:
    raise Exception("PDF_PATH is not set in the environment variables..")