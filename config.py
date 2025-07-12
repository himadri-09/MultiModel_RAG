import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY =os.getenv("OPENROUTER_API_KEY")
# Model Configuration
EMBEDDING_MODEL = "models/text-embedding-004"  
GEMINI_MODEL = "gemini-1.5-flash"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = "2025-01-01-preview" 


# Storage Configuration
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
VECTORS_DIR = DATA_DIR / "vectors"
METADATA_DIR = DATA_DIR / "metadata"

# Create directories
for dir_path in [DATA_DIR, IMAGES_DIR, VECTORS_DIR, METADATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# Processing Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

