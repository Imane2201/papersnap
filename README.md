# PaperSnap 📄✨

**PaperSnap** is an AI-powered tool that takes a research paper (either uploaded as a PDF or via an ArXiv link) and generates a **clean, structured 1-page Markdown summary**. It extracts the title, authors, key ideas, contributions, limitations, and generates a citation-ready output.

## Features

- 📄 **PDF Processing**: Extract text and metadata from PDF files
- 🔗 **ArXiv Integration**: Download and process papers directly from ArXiv
- 🤖 **AI-Powered Summarization**: Uses LangChain and OpenAI/Azure OpenAI for intelligent summarization
- 📝 **Structured Output**: Generates clean, organized Markdown summaries
- 🔍 **Key Insights**: Extracts contributions, methodology, results, and limitations
- 📚 **Citation Generation**: Automatic APA-style citation generation
- 🎯 **CLI Interface**: Easy-to-use command-line interface
- 🌐 **Multiple Formats**: Output in Markdown or HTML format

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key or Azure OpenAI credentials

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Imane2201/papersnap.git
cd papersnap

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the setup command:
   ```bash
   python main.py setup
   ```

## Usage

### Command Line Interface

#### Summarize a PDF file:
```bash
python main.py summarize --pdf path/to/paper.pdf
```

#### Summarize an ArXiv paper:
```bash
python main.py summarize --arxiv https://arxiv.org/abs/2301.00001
# or
python main.py summarize --arxiv 2301.00001
```

#### Specify output format and location:
```bash
python main.py summarize --arxiv 2301.00001 --format markdown --output my_summary.md
```

#### Enable verbose logging:
```bash
python main.py summarize --pdf paper.pdf --verbose
```

### Python API

```python
from papersnap import PDFParser, ArxivHandler, PaperSummarizer, MarkdownGenerator

# Initialize components
pdf_parser = PDFParser()
arxiv_handler = ArxivHandler()
summarizer = PaperSummarizer()
markdown_generator = MarkdownGenerator()

# Process an ArXiv paper
pdf_path = arxiv_handler.download_paper("2301.00001")
paper_content = pdf_parser.extract_text(pdf_path)
paper_metadata = arxiv_handler.get_paper_metadata("2301.00001")

# Generate summary
summary = summarizer.summarize_paper(paper_content, paper_metadata)
markdown_output = markdown_generator.generate_markdown(summary, paper_metadata)

print(markdown_output)
```

## Configuration

PaperSnap can be configured through environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Optional |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Optional |
| `MODEL_NAME` | Model to use | `gpt-3.5-turbo` |
| `MODEL_TEMPERATURE` | Model temperature | `0.1` |
| `MAX_TOKENS` | Maximum tokens per request | `4000` |
| `CHUNK_SIZE` | Text chunk size | `2000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `PDF_MAX_PAGES` | Maximum pages to process | `50` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Output Structure

PaperSnap generates structured summaries with the following sections:

- **Title and Authors**: Paper title and author list
- **Abstract**: Original or AI-generated abstract
- **Key Contributions**: Main contributions and novel aspects
- **Methodology**: Research methods and approach
- **Results**: Key findings and outcomes
- **Limitations**: Acknowledged limitations
- **Future Work**: Suggested future research directions
- **Overall Summary**: Comprehensive overview
- **Technical Details**: Datasets, tools, and frameworks used
- **Citation**: APA-style citation
- **Paper Information**: Metadata and links

## Example Output

```markdown
# Attention Is All You Need

## Authors
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin

## Abstract
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely...

## Key Contributions
- Introduction of the Transformer architecture based entirely on attention mechanisms
- Elimination of recurrence and convolutions in sequence modeling
- Achievement of state-of-the-art results on machine translation tasks
- Demonstration of superior parallelization capabilities

## Methodology
The Transformer uses self-attention mechanisms to process sequences...

[... rest of the summary ...]
```

## Project Structure

```
papersnap/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # This file
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── papersnap/             # Main package
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── pdf_parser.py      # PDF processing
│   ├── arxiv_handler.py   # ArXiv integration
│   ├── summarizer.py      # AI summarization
│   └── markdown_generator.py # Output generation
├── output/                # Generated summaries
├── logs/                  # Log files
├── tests/                 # Test files
└── examples/              # Example files
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style

This project follows PEP 8 coding standards. Use `black` for code formatting:

```bash
black papersnap/
```

## Limitations

- Requires OpenAI API key (paid service)
- PDF parsing quality depends on document structure
- Large papers may be truncated to fit token limits
- Non-English papers may have reduced accuracy

## Future Enhancements

- [ ] Streamlit web interface
- [ ] Export to Notion and other platforms
- [ ] Voice summary narration
- [ ] Multi-language support
- [ ] Batch processing capabilities
- [ ] Custom summary templates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- PDF processing powered by [PyMuPDF](https://pymupdf.readthedocs.io/)
- ArXiv integration using [arxiv-py](https://github.com/lukasschwab/arxiv.py)
- OpenAI GPT models for summarization

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/papersnap/issues) page
2. Create a new issue if needed
3. Join our [Discussions](https://github.com/yourusername/papersnap/discussions)

---

Made with ❤️ by Imane LABBASSI