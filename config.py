import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure your .env has this value

# FAISS Index Configuration
PERSIST_DIRECTORY = r"C:\Users\Ms-Echo\OneDrive\Dokumente\Coding Assistant\faiss_db"  # Updated path to FAISS directory
LOG_FILE = r"C:\Users\Ms-Echo\OneDrive\Dokumente\Coding Assistant\logfile.log"  # Path for logging

# HuggingFace Model Configuration
DEFAULT_MODEL_CHECKPOINT = "MBZUAI/LaMini-T5-738M"  # Model for loading the LLM

# FAISS Settings
FAISS_SETTINGS = {
    "persist_directory": PERSIST_DIRECTORY,  # Directory where FAISS stores index files
    "index_file": "index.faiss",  # Name of the FAISS index file
    "use_gpu": False,  # Set True if GPU should be used for FAISS (if supported)
    "embedding_model_name": "all-MiniLM-L6-v2"  # Embeddings model used in FAISS
}

# Other Configurations (add more as needed)
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == 'true'  # Control debug output
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 256))  # Define max tokens for LLM
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))  # Sampling temperature for generation
TOP_P = float(os.getenv("TOP_P", 0.95))  # Top-p sampling for diversity
