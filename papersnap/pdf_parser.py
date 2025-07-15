"""
PDF Parser module for PaperSnap
Handles PDF text extraction and metadata parsing using PyMuPDF
"""

import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .config import Config

logger = logging.getLogger(__name__)

class PDFParser:
    """PDF parser for extracting text and metadata from research papers"""
    
    def __init__(self):
        self.max_pages = Config.PDF_MAX_PAGES
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            # Limit pages to prevent processing extremely long documents
            max_pages = min(len(doc), self.max_pages)
            
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean up the text
                text = self._clean_text(text)
                if text.strip():
                    text_content.append(text)
            
            doc.close()
            full_text = "\n\n".join(text_content)
            
            logger.info(f"Extracted {len(full_text)} characters from {max_pages} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Extract additional information from first page
            first_page = doc.load_page(0)
            first_page_text = first_page.get_text()
            
            # Try to extract title from metadata or first page
            title = metadata.get("title", "")
            if not title:
                title = self._extract_title_from_text(first_page_text)
            
            # Try to extract authors
            authors = self._extract_authors_from_text(first_page_text)
            
            # Create metadata dictionary
            paper_metadata = {
                "title": title or "Unknown Title",
                "authors": authors,
                "pdf_path": str(pdf_path),
                "pages": len(doc),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "processed_date": datetime.now().isoformat(),
            }
            
            doc.close()
            logger.info(f"Extracted metadata for paper: {paper_metadata['title']}")
            return paper_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            return {
                "title": "Unknown Title",
                "authors": [],
                "pdf_path": str(pdf_path),
                "pages": 0,
                "processed_date": datetime.now().isoformat(),
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Join lines and normalize spacing
        cleaned_text = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from the first page text"""
        lines = text.split('\n')
        
        # Look for title patterns (usually in the first few lines)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # Skip very short lines or lines with common header patterns
            if (len(line) > 20 and 
                not line.lower().startswith(('abstract', 'introduction', 'arxiv:', 'doi:', 'http')) and
                not line.isdigit() and
                not line.lower().startswith('keywords')):
                return line
        
        return ""
    
    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Extract authors from the first page text"""
        lines = text.split('\n')
        authors = []
        
        # Look for author patterns
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            
            # Skip title and other non-author lines
            if (len(line) > 5 and len(line) < 200 and
                not line.lower().startswith(('abstract', 'introduction', 'arxiv:', 'doi:', 'http', 'keywords')) and
                not line.isdigit()):
                
                # Check if line contains potential author names
                if self._looks_like_authors(line):
                    # Split by common separators
                    import re
                    potential_authors = re.split(r'[,;]|(?:\s+and\s+)', line)
                    
                    for author in potential_authors:
                        author = author.strip()
                        if author and len(author) > 2:
                            authors.append(author)
                    
                    if authors:
                        break
        
        return authors[:10]  # Limit to reasonable number of authors
    
    def _looks_like_authors(self, text: str) -> bool:
        """Check if text looks like author names"""
        # Simple heuristic: contains capital letters and common name patterns
        import re
        
        # Check for patterns that suggest author names
        patterns = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
            r'[A-Z]\.\s*[A-Z][a-z]+',      # F. Last
            r'[A-Z][a-z]+\s+[A-Z]\.',      # First L.
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks for processing"""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        if overlap is None:
            overlap = Config.CHUNK_OVERLAP
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks