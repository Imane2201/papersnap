#!/usr/bin/env python3
"""
Example usage of PaperSnap library
Demonstrates how to use PaperSnap programmatically
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from papersnap import PDFParser, ArxivHandler, PaperSummarizer, MarkdownGenerator
from papersnap.config import Config

def example_arxiv_summarization():
    """Example: Summarize an ArXiv paper"""
    print("🔍 Example: Summarizing ArXiv paper...")
    
    # ArXiv paper ID (famous Transformer paper)
    arxiv_id = "1706.03762"  # Attention Is All You Need
    
    try:
        # Initialize components
        arxiv_handler = ArxivHandler()
        pdf_parser = PDFParser()
        summarizer = PaperSummarizer()
        markdown_generator = MarkdownGenerator()
        
        # Download and get metadata
        print(f"📥 Downloading paper: {arxiv_id}")
        pdf_path = arxiv_handler.download_paper(arxiv_id)
        paper_metadata = arxiv_handler.get_paper_metadata(arxiv_id)
        
        # Extract text
        print("📄 Extracting text from PDF...")
        paper_content = pdf_parser.extract_text(pdf_path)
        
        # Generate summary
        print("🤖 Generating AI summary...")
        summary = summarizer.summarize_paper(paper_content, paper_metadata)
        
        # Generate markdown
        print("📝 Generating markdown output...")
        markdown_output = markdown_generator.generate_markdown(summary, paper_metadata)
        
        # Save output
        output_path = Config.OUTPUT_DIR / f"attention_is_all_you_need.md"
        output_path.write_text(markdown_output, encoding='utf-8')
        
        print(f"✅ Summary saved to: {output_path}")
        print(f"📊 Summary length: {len(markdown_output)} characters")
        
        # Cleanup
        arxiv_handler.cleanup_temp_files()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_pdf_summarization():
    """Example: Summarize a local PDF file"""
    print("\n📄 Example: Summarizing local PDF...")
    
    # This would work with a real PDF file
    pdf_path = Path("sample_paper.pdf")
    
    if not pdf_path.exists():
        print("⚠️  Sample PDF not found. Skipping this example.")
        return
    
    try:
        # Initialize components
        pdf_parser = PDFParser()
        summarizer = PaperSummarizer()
        markdown_generator = MarkdownGenerator()
        
        # Extract text and metadata
        print("📄 Extracting text from PDF...")
        paper_content = pdf_parser.extract_text(pdf_path)
        paper_metadata = pdf_parser.extract_metadata(pdf_path)
        
        # Generate summary
        print("🤖 Generating AI summary...")
        summary = summarizer.summarize_paper(paper_content, paper_metadata)
        
        # Generate markdown
        print("📝 Generating markdown output...")
        markdown_output = markdown_generator.generate_markdown(summary, paper_metadata)
        
        # Save output
        output_path = Config.OUTPUT_DIR / f"local_paper_summary.md"
        output_path.write_text(markdown_output, encoding='utf-8')
        
        print(f"✅ Summary saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_batch_processing():
    """Example: Process multiple papers"""
    print("\n🔄 Example: Batch processing multiple papers...")
    
    # List of ArXiv paper IDs
    paper_ids = [
        "1706.03762",  # Attention Is All You Need
        "1801.04381",  # ImageNet Classification with Deep Convolutional Neural Networks
        "1412.6980",   # Adam: A Method for Stochastic Optimization
    ]
    
    # Initialize components
    arxiv_handler = ArxivHandler()
    pdf_parser = PDFParser()
    summarizer = PaperSummarizer()
    markdown_generator = MarkdownGenerator()
    
    for i, paper_id in enumerate(paper_ids[:1]):  # Process only first paper in example
        try:
            print(f"📄 Processing paper {i+1}/{len(paper_ids)}: {paper_id}")
            
            # Download and process
            pdf_path = arxiv_handler.download_paper(paper_id)
            paper_metadata = arxiv_handler.get_paper_metadata(paper_id)
            paper_content = pdf_parser.extract_text(pdf_path)
            
            # Generate summary
            summary = summarizer.summarize_paper(paper_content, paper_metadata)
            markdown_output = markdown_generator.generate_markdown(summary, paper_metadata)
            
            # Save with clean filename
            title = paper_metadata.get('title', f'paper_{paper_id}')
            clean_title = ''.join(c for c in title if c.isalnum() or c in ('-', '_', ' '))[:50]
            clean_title = clean_title.replace(' ', '_')
            
            output_path = Config.OUTPUT_DIR / f"{clean_title}.md"
            output_path.write_text(markdown_output, encoding='utf-8')
            
            print(f"✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {paper_id}: {e}")
    
    # Cleanup
    arxiv_handler.cleanup_temp_files()

def example_custom_configuration():
    """Example: Using custom configuration"""
    print("\n⚙️  Example: Custom configuration...")
    
    # Show current configuration
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Temperature: {Config.MODEL_TEMPERATURE}")
    print(f"Max Tokens: {Config.MAX_TOKENS}")
    print(f"Chunk Size: {Config.CHUNK_SIZE}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"Azure OpenAI: {Config.is_azure_openai()}")

def example_error_handling():
    """Example: Error handling"""
    print("\n🛡️  Example: Error handling...")
    
    try:
        # Try to process invalid ArXiv ID
        arxiv_handler = ArxivHandler()
        arxiv_handler.download_paper("invalid-id")
    except Exception as e:
        print(f"✅ Caught expected error: {e}")
    
    try:
        # Try to process non-existent PDF
        pdf_parser = PDFParser()
        pdf_parser.extract_text(Path("nonexistent.pdf"))
    except Exception as e:
        print(f"✅ Caught expected error: {e}")

def main():
    """Run all examples"""
    print("🚀 PaperSnap Library Examples")
    print("=" * 50)
    
    # Check if API key is configured
    if not Config.OPENAI_API_KEY and not Config.AZURE_OPENAI_API_KEY:
        print("⚠️  Warning: No API key configured. Set OPENAI_API_KEY or Azure OpenAI credentials.")
        print("Examples will show errors without valid API keys.")
    
    # Run examples
    example_custom_configuration()
    example_error_handling()
    
    # Only run AI examples if API key is available
    if Config.OPENAI_API_KEY or Config.AZURE_OPENAI_API_KEY:
        try:
            Config.validate()
            example_arxiv_summarization()
            example_pdf_summarization()
            example_batch_processing()
        except ValueError as e:
            print(f"⚠️  Configuration error: {e}")
    else:
        print("\n⚠️  Skipping AI examples due to missing API key.")
    
    print("\n🎉 Examples completed!")

if __name__ == "__main__":
    main()