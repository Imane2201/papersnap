"""
PaperSnap - AI-powered research paper summarization tool

This package provides tools for extracting and summarizing research papers
from PDF files or ArXiv links into clean, structured Markdown summaries.
"""

__version__ = "1.0.0"
__author__ = "PaperSnap Team"
__email__ = "contact@papersnap.dev"

from .cli import main
from .pdf_parser import PDFParser
from .arxiv_handler import ArxivHandler
from .summarizer import PaperSummarizer
from .markdown_generator import MarkdownGenerator

__all__ = [
    "main",
    "PDFParser",
    "ArxivHandler", 
    "PaperSummarizer",
    "MarkdownGenerator",
]