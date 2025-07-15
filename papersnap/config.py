"""
Configuration module for PaperSnap
Handles environment variables, settings, and configuration management
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for PaperSnap"""
    
    # OpenAI/Azure OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # File Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    TEMP_DIR: Path = PROJECT_ROOT / "temp"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ArXiv Configuration
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "1"))
    ARXIV_SORT_BY: str = os.getenv("ARXIV_SORT_BY", "relevance")
    
    # PDF Processing
    PDF_MAX_PAGES: int = int(os.getenv("PDF_MAX_PAGES", "50"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY and not cls.AZURE_OPENAI_API_KEY:
            raise ValueError(
                "Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set"
            )
        
        if cls.AZURE_OPENAI_API_KEY and not cls.AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT must be set when using Azure OpenAI"
            )
        
        return True
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories"""
        for directory in [cls.OUTPUT_DIR, cls.LOGS_DIR, cls.TEMP_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def is_azure_openai(cls) -> bool:
        """Check if Azure OpenAI is configured"""
        return bool(cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_ENDPOINT)

# Create directories on import
Config.setup_directories()