"""
Command Line Interface for PaperSnap
Handles CLI arguments and orchestrates the paper summarization process
"""

import logging
import click
from pathlib import Path
from typing import Optional

from .config import Config
from .pdf_parser import PDFParser
from .arxiv_handler import ArxivHandler
from .summarizer import PaperSummarizer
from .markdown_generator import MarkdownGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / "papersnap.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """PaperSnap - AI-powered research paper summarization tool"""
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        click.echo(f"Error: {e}", err=True)
        exit(1)

@cli.command()
@click.option(
    "--pdf", 
    type=click.Path(exists=True, path_type=Path),
    help="Path to PDF file to summarize"
)
@click.option(
    "--arxiv", 
    type=str,
    help="ArXiv URL or paper ID to summarize"
)
@click.option(
    "--output", 
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: output/{title}.md)"
)
@click.option(
    "--format", 
    type=click.Choice(['markdown', 'html']),
    default='markdown',
    help="Output format"
)
@click.option(
    "--verbose", 
    is_flag=True,
    help="Enable verbose logging"
)
def summarize(pdf: Optional[Path], arxiv: Optional[str], output: Optional[Path], 
              format: str, verbose: bool):
    """Summarize a research paper from PDF or ArXiv"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not pdf and not arxiv:
        click.echo("Error: Either --pdf or --arxiv must be specified", err=True)
        exit(1)
    
    if pdf and arxiv:
        click.echo("Error: Cannot specify both --pdf and --arxiv", err=True)
        exit(1)
    
    try:
        # Initialize components
        pdf_parser = PDFParser()
        arxiv_handler = ArxivHandler()
        summarizer = PaperSummarizer()
        markdown_generator = MarkdownGenerator()
        
        # Get PDF content
        if pdf:
            logger.info(f"Processing PDF: {pdf}")
            paper_content = pdf_parser.extract_text(pdf)
            paper_metadata = pdf_parser.extract_metadata(pdf)
        else:
            logger.info(f"Processing ArXiv paper: {arxiv}")
            pdf_path = arxiv_handler.download_paper(arxiv)
            paper_content = pdf_parser.extract_text(pdf_path)
            paper_metadata = arxiv_handler.get_paper_metadata(arxiv)
        
        # Summarize the paper
        logger.info("Generating summary...")
        summary = summarizer.summarize_paper(paper_content, paper_metadata)
        
        # Generate output
        if format == 'markdown':
            output_content = markdown_generator.generate_markdown(summary, paper_metadata)
            file_extension = '.md'
        else:
            output_content = markdown_generator.generate_html(summary, paper_metadata)
            file_extension = '.html'
        
        # Determine output path
        if output is None:
            title = paper_metadata.get('title', 'summary').replace(' ', '_')
            # Clean title for filename
            title = ''.join(c for c in title if c.isalnum() or c in ('-', '_'))[:50]
            output = Config.OUTPUT_DIR / f"{title}{file_extension}"
        else:
            # If output is specified but it's a relative path, make it relative to output directory
            if not output.is_absolute():
                output = Config.OUTPUT_DIR / output
        
        # Write output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_content, encoding='utf-8')
        
        logger.info(f"Summary saved to: {output}")
        click.echo(f"Summary saved to: {output}")
        
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        click.echo(f"Error: {e}", err=True)
        exit(1)

@cli.command()
def setup():
    """Set up PaperSnap configuration"""
    click.echo("Setting up PaperSnap...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        click.echo("Found existing .env file")
    else:
        click.echo("Creating .env file...")
        env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Azure OpenAI Configuration (alternative to OpenAI)
# AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Model Configuration
MODEL_NAME=gpt-3.5-turbo
MODEL_TEMPERATURE=0.1
MAX_TOKENS=4000

# Chunking Configuration
CHUNK_SIZE=2000
CHUNK_OVERLAP=200

# Logging
LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        click.echo("Created .env file. Please update it with your API keys.")
    
    # Create directories
    Config.setup_directories()
    click.echo("Setup complete!")

@cli.command()
def version():
    """Show version information"""
    click.echo("PaperSnap v1.0.0")
    click.echo("AI-powered research paper summarization tool")

def main():
    """Main entry point for the CLI"""
    cli()