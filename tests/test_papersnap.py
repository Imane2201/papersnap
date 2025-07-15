"""
Test cases for PaperSnap modules
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from papersnap.config import Config
from papersnap.pdf_parser import PDFParser
from papersnap.arxiv_handler import ArxivHandler
from papersnap.summarizer import PaperSummarizer
from papersnap.markdown_generator import MarkdownGenerator

class TestConfig:
    """Test configuration management"""
    
    def test_config_validation_missing_api_key(self):
        """Test that config validation fails when no API key is provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set"):
                Config.validate()
    
    def test_config_validation_with_openai_key(self):
        """Test that config validation passes with OpenAI key"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            assert Config.validate() is True
    
    def test_is_azure_openai(self):
        """Test Azure OpenAI detection"""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_API_KEY': 'test_key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com'
        }):
            assert Config.is_azure_openai() is True

class TestPDFParser:
    """Test PDF parsing functionality"""
    
    def test_init(self):
        """Test PDFParser initialization"""
        parser = PDFParser()
        assert parser.max_pages == Config.PDF_MAX_PAGES
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        parser = PDFParser()
        dirty_text = "This is    a test\n\nwith   multiple   spaces\n\n\n"
        clean_text = parser._clean_text(dirty_text)
        assert clean_text == "This is a test with multiple spaces"
    
    def test_looks_like_authors(self):
        """Test author name detection"""
        parser = PDFParser()
        assert parser._looks_like_authors("John Smith") is True
        assert parser._looks_like_authors("J. Smith") is True
        assert parser._looks_like_authors("John S.") is True
        assert parser._looks_like_authors("123456") is False
    
    def test_chunk_text(self):
        """Test text chunking"""
        parser = PDFParser()
        text = "This is a test sentence. " * 100
        chunks = parser.chunk_text(text, chunk_size=10, overlap=2)
        assert len(chunks) > 1
        assert isinstance(chunks[0], str)

class TestArxivHandler:
    """Test ArXiv handling functionality"""
    
    def test_init(self):
        """Test ArxivHandler initialization"""
        handler = ArxivHandler()
        assert handler.temp_dir == Config.TEMP_DIR
        assert handler.max_results == Config.ARXIV_MAX_RESULTS
    
    def test_extract_arxiv_id_from_url(self):
        """Test ArXiv ID extraction from URLs"""
        handler = ArxivHandler()
        
        # Test different URL formats
        assert handler._extract_arxiv_id("https://arxiv.org/abs/2301.00001") == "2301.00001"
        assert handler._extract_arxiv_id("https://arxiv.org/pdf/2301.00001.pdf") == "2301.00001"
        assert handler._extract_arxiv_id("arxiv:2301.00001") == "2301.00001"
        assert handler._extract_arxiv_id("2301.00001") == "2301.00001"
    
    def test_is_valid_arxiv_id(self):
        """Test ArXiv ID validation"""
        handler = ArxivHandler()
        
        # Valid IDs
        assert handler._is_valid_arxiv_id("2301.00001") is True
        assert handler._is_valid_arxiv_id("2301.00001v1") is True
        assert handler._is_valid_arxiv_id("math.CO/0701001") is True
        
        # Invalid IDs
        assert handler._is_valid_arxiv_id("invalid-id") is False
        assert handler._is_valid_arxiv_id("") is False

class TestMarkdownGenerator:
    """Test markdown generation functionality"""
    
    def test_init(self):
        """Test MarkdownGenerator initialization"""
        generator = MarkdownGenerator()
        assert generator.template is not None
    
    def test_format_authors(self):
        """Test author formatting"""
        generator = MarkdownGenerator()
        
        # Test single author
        assert generator._format_authors(["John Smith"]) == "John Smith"
        
        # Test two authors
        assert generator._format_authors(["John Smith", "Jane Doe"]) == "John Smith and Jane Doe"
        
        # Test multiple authors
        authors = ["John Smith", "Jane Doe", "Bob Johnson"]
        expected = "John Smith, Jane Doe, and Bob Johnson"
        assert generator._format_authors(authors) == expected
        
        # Test empty list
        assert generator._format_authors([]) == "*Authors not specified*"
    
    def test_format_list_items(self):
        """Test list item formatting"""
        generator = MarkdownGenerator()
        
        items = ["First contribution", "Second contribution"]
        result = generator._format_list_items(items)
        assert "- First contribution" in result
        assert "- Second contribution" in result
        
        # Test empty list
        assert generator._format_list_items([]) == "*No specific contributions identified*"
    
    def test_generate_citation(self):
        """Test citation generation"""
        generator = MarkdownGenerator()
        
        metadata = {
            "title": "Test Paper",
            "authors": ["John Smith", "Jane Doe"],
            "published": "2023-01-01T00:00:00Z",
            "arxiv_id": "2301.00001"
        }
        
        citation = generator._generate_citation(metadata)
        assert "John Smith & Jane Doe" in citation
        assert "(2023)" in citation
        assert "Test Paper" in citation
        assert "arXiv:2301.00001" in citation

class TestPaperSummarizer:
    """Test paper summarization functionality"""
    
    @patch('papersnap.summarizer.ChatOpenAI')
    def test_init_with_openai(self, mock_openai):
        """Test PaperSummarizer initialization with OpenAI"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            summarizer = PaperSummarizer()
            assert summarizer.config == Config
            assert summarizer.text_splitter is not None
    
    @patch('papersnap.summarizer.AzureChatOpenAI')
    def test_init_with_azure(self, mock_azure):
        """Test PaperSummarizer initialization with Azure OpenAI"""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_API_KEY': 'test_key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com'
        }):
            summarizer = PaperSummarizer()
            assert summarizer.config == Config
    
    @patch('papersnap.summarizer.ChatOpenAI')
    def test_combine_relevant_chunks(self, mock_openai):
        """Test combining relevant chunks"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            summarizer = PaperSummarizer()
            
            # Mock documents
            from langchain.docstore.document import Document
            docs = [
                Document(page_content="This is about methodology and approach"),
                Document(page_content="This is about results and findings"),
                Document(page_content="This is about something else")
            ]
            
            # Test combining chunks with relevant keywords
            combined = summarizer._combine_relevant_chunks(docs, ["method", "approach"])
            assert "methodology and approach" in combined
            assert len(combined) > 0

class TestIntegration:
    """Integration tests"""
    
    def test_project_structure(self):
        """Test that all required files exist"""
        project_root = Path(__file__).parent.parent
        
        # Check main files
        assert (project_root / "main.py").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "setup.py").exists()
        assert (project_root / "README.md").exists()
        
        # Check package structure
        package_dir = project_root / "papersnap"
        assert package_dir.exists()
        assert (package_dir / "__init__.py").exists()
        assert (package_dir / "cli.py").exists()
        assert (package_dir / "config.py").exists()
        assert (package_dir / "pdf_parser.py").exists()
        assert (package_dir / "arxiv_handler.py").exists()
        assert (package_dir / "summarizer.py").exists()
        assert (package_dir / "markdown_generator.py").exists()
    
    def test_imports(self):
        """Test that all modules can be imported"""
        from papersnap import (
            PDFParser, ArxivHandler, PaperSummarizer, 
            MarkdownGenerator, main
        )
        
        assert PDFParser is not None
        assert ArxivHandler is not None
        assert PaperSummarizer is not None
        assert MarkdownGenerator is not None
        assert main is not None

# Fixtures for testing
@pytest.fixture
def sample_paper_metadata():
    """Sample paper metadata for testing"""
    return {
        "title": "Test Paper: A Comprehensive Analysis",
        "authors": ["John Smith", "Jane Doe", "Bob Johnson"],
        "arxiv_id": "2301.00001",
        "arxiv_url": "https://arxiv.org/abs/2301.00001",
        "abstract": "This is a test abstract for the paper.",
        "published": "2023-01-01T00:00:00Z",
        "categories": ["cs.AI", "cs.LG"],
        "primary_category": "cs.AI"
    }

@pytest.fixture
def sample_summary_data():
    """Sample summary data for testing"""
    return {
        "title": "Test Paper: A Comprehensive Analysis",
        "authors": ["John Smith", "Jane Doe"],
        "abstract": "This is a test abstract.",
        "key_contributions": [
            "First major contribution",
            "Second important finding",
            "Third novel approach"
        ],
        "methodology": "The authors used a novel approach...",
        "results": "The results show significant improvements...",
        "limitations": "The main limitations include...",
        "future_work": "Future work should focus on...",
        "overall_summary": "This paper presents a comprehensive analysis...",
        "technical_details": {
            "datasets": "MNIST, CIFAR-10",
            "tools": "PyTorch, TensorFlow"
        }
    }

if __name__ == "__main__":
    pytest.main([__file__])