import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "models/text-embedding-004"  # Gemini's free embedding model
GEMINI_MODEL = "gemini-1.5-flash"

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