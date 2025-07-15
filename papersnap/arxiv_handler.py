"""
ArXiv Handler module for PaperSnap
Handles downloading and metadata extraction from ArXiv papers
"""

import logging
import re
import requests
import arxiv
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import Config

logger = logging.getLogger(__name__)

class ArxivHandler:
    """Handler for ArXiv paper downloading and metadata extraction"""
    
    def __init__(self):
        self.temp_dir = Config.TEMP_DIR
        self.max_results = Config.ARXIV_MAX_RESULTS
        self.sort_by = Config.ARXIV_SORT_BY
        self.temp_dir.mkdir(exist_ok=True)
    
    def download_paper(self, arxiv_input: str) -> Path:
        """Download paper from ArXiv URL or ID"""
        try:
            # Extract ArXiv ID from URL or use as-is
            arxiv_id = self._extract_arxiv_id(arxiv_input)
            logger.info(f"Downloading ArXiv paper: {arxiv_id}")
            
            # Search for the paper
            client = arxiv.Client()
            search = arxiv.Search(
                query=f"id:{arxiv_id}",
                max_results=1,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = list(client.results(search))
            
            if not results:
                raise Exception(f"Paper not found: {arxiv_id}")
            
            paper = results[0]
            
            # Download PDF
            pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
            pdf_path = self.temp_dir / pdf_filename
            
            paper.download_pdf(dirpath=str(self.temp_dir), filename=pdf_filename)
            
            logger.info(f"Downloaded paper to: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading ArXiv paper: {e}")
            raise Exception(f"Failed to download ArXiv paper: {e}")
    
    def get_paper_metadata(self, arxiv_input: str) -> Dict[str, Any]:
        """Get metadata for ArXiv paper"""
        try:
            arxiv_id = self._extract_arxiv_id(arxiv_input)
            
            # Search for the paper
            client = arxiv.Client()
            search = arxiv.Search(
                query=f"id:{arxiv_id}",
                max_results=1,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = list(client.results(search))
            
            if not results:
                raise Exception(f"Paper not found: {arxiv_id}")
            
            paper = results[0]
            
            # Extract metadata
            metadata = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "arxiv_id": arxiv_id,
                "arxiv_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "abstract": paper.summary,
                "published": paper.published.isoformat() if paper.published else "",
                "updated": paper.updated.isoformat() if paper.updated else "",
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "comment": paper.comment or "",
                "journal_ref": paper.journal_ref or "",
                "doi": paper.doi or "",
                "processed_date": datetime.now().isoformat(),
            }
            
            logger.info(f"Retrieved metadata for paper: {metadata['title']}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting ArXiv metadata: {e}")
            # Return basic metadata if extraction fails
            return {
                "title": "Unknown ArXiv Paper",
                "authors": [],
                "arxiv_id": arxiv_input,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_input}",
                "abstract": "",
                "categories": [],
                "processed_date": datetime.now().isoformat(),
            }
    
    def _extract_arxiv_id(self, arxiv_input: str) -> str:
        """Extract ArXiv ID from URL or return as-is if already an ID"""
        # Remove whitespace
        arxiv_input = arxiv_input.strip()
        
        # Handle various ArXiv URL formats
        patterns = [
            r'arxiv\.org/abs/([^/?]+)',
            r'arxiv\.org/pdf/([^/?]+)',
            r'arxiv\.org/(?:abs|pdf)/([^/?]+)\.pdf',
            r'arxiv:([^/?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, arxiv_input, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no URL pattern found, assume it's already an ID
        # Clean up common formatting
        arxiv_id = re.sub(r'[^\w\.-]', '', arxiv_input)
        
        # Validate ArXiv ID format
        if self._is_valid_arxiv_id(arxiv_id):
            return arxiv_id
        
        raise ValueError(f"Invalid ArXiv ID or URL: {arxiv_input}")
    
    def _is_valid_arxiv_id(self, arxiv_id: str) -> bool:
        """Validate ArXiv ID format"""
        # ArXiv ID patterns
        patterns = [
            r'^\d{4}\.\d{4,5}$',  # New format: 2107.12345
            r'^\d{4}\.\d{4,5}v\d+$',  # New format with version: 2107.12345v1
            r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$',  # Old format: math.CO/0701001
            r'^[a-z-]+(\.[A-Z]{2})?/\d{7}v\d+$',  # Old format with version
        ]
        
        for pattern in patterns:
            if re.match(pattern, arxiv_id):
                return True
        
        return False
    
    def search_papers(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search for papers on ArXiv"""
        if max_results is None:
            max_results = self.max_results
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in client.results(search):
                paper_info = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "arxiv_id": paper.get_short_id(),
                    "arxiv_url": paper.entry_id,
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else "",
                    "categories": paper.categories,
                    "primary_category": paper.primary_category,
                }
                results.append(paper_info)
            
            logger.info(f"Found {len(results)} papers for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def cleanup_temp_files(self):
        """Clean up temporary downloaded files"""
        try:
            for file in self.temp_dir.glob("*.pdf"):
                file.unlink()
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")