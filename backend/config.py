import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file (from project root)
# Get the directory where this file is located (backend/)
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of backend/)
_project_root = os.path.dirname(_current_dir)
# Load .env from project root
load_dotenv(os.path.join(_project_root, ".env"))

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Moonshot AI (Kimi) API settings
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    MOONSHOT_BASE_URL: str = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    MOONSHOT_MODEL: str = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


