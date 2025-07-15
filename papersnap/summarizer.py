"""
Summarizer module for PaperSnap
Handles AI-powered summarization of research papers using LangChain and OpenAI
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from .config import Config

logger = logging.getLogger(__name__)

class PaperSummarizer:
    """AI-powered paper summarization using LangChain and OpenAI"""
    
    def __init__(self):
        self.config = Config
        self.llm = self._initialize_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
        
    def _initialize_llm(self):
        """Initialize the language model (OpenAI or Azure OpenAI)"""
        try:
            if self.config.is_azure_openai():
                logger.info("Initializing Azure OpenAI")
                return AzureChatOpenAI(
                    azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                    api_key=self.config.AZURE_OPENAI_API_KEY,
                    api_version=self.config.AZURE_OPENAI_API_VERSION,
                    deployment_name=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                    temperature=self.config.MODEL_TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                )
            else:
                logger.info("Initializing OpenAI")
                return ChatOpenAI(
                    api_key=self.config.OPENAI_API_KEY,
                    model_name=self.config.MODEL_NAME,
                    temperature=self.config.MODEL_TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise Exception(f"Failed to initialize language model: {e}")
    
    def summarize_paper(self, paper_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured summary of the research paper"""
        try:
            logger.info("Starting paper summarization")
            
            # Split text into chunks
            documents = self.text_splitter.create_documents([paper_content])
            logger.info(f"Split paper into {len(documents)} chunks")
            
            # Generate different types of summaries
            summary_data = {
                "title": metadata.get("title", "Unknown Title"),
                "authors": metadata.get("authors", []),
                "abstract": self._extract_or_generate_abstract(paper_content, metadata),
                "key_contributions": self._extract_key_contributions(documents),
                "methodology": self._extract_methodology(documents),
                "results": self._extract_results(documents),
                "limitations": self._extract_limitations(documents),
                "future_work": self._extract_future_work(documents),
                "overall_summary": self._generate_overall_summary(documents),
                "technical_details": self._extract_technical_details(documents),
                "metadata": metadata,
                "processed_date": datetime.now().isoformat(),
            }
            
            logger.info("Paper summarization completed")
            return summary_data
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise Exception(f"Failed to summarize paper: {e}")
    
    def _extract_or_generate_abstract(self, paper_content: str, metadata: Dict[str, Any]) -> str:
        """Extract abstract from metadata or generate one from content"""
        # Check if abstract is available in metadata (ArXiv papers)
        if metadata.get("abstract"):
            return metadata["abstract"]
        
        # Try to extract abstract from paper content
        lines = paper_content.split('\n')
        abstract_start = -1
        abstract_end = -1
        
        for i, line in enumerate(lines):
            if 'abstract' in line.lower() and abstract_start == -1:
                abstract_start = i
            elif abstract_start != -1 and ('introduction' in line.lower() or 
                                         'keywords' in line.lower() or 
                                         len(line.strip()) == 0):
                abstract_end = i
                break
        
        if abstract_start != -1 and abstract_end != -1:
            abstract_lines = lines[abstract_start:abstract_end]
            abstract = ' '.join(abstract_lines).replace('abstract', '').strip()
            if len(abstract) > 50:  # Reasonable abstract length
                return abstract
        
        # Generate abstract if not found
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Based on the following research paper content, write a concise abstract (2-3 sentences) that summarizes the main contribution, methodology, and key findings:

            {text}

            Abstract:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Use first few chunks for abstract generation
        content_sample = ' '.join([doc.page_content for doc in 
                                  self.text_splitter.create_documents([paper_content])[:3]])
        
        result = chain.run(text=content_sample[:4000])
        return result.strip()
    
    def _extract_key_contributions(self, documents: List[Document]) -> List[str]:
        """Extract key contributions from the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following research paper text, identify and list the key contributions or novel aspects of this work. 
            Return 3-5 bullet points focusing on what makes this research unique or valuable:

            {text}

            Key Contributions:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Combine relevant chunks
        combined_text = self._combine_relevant_chunks(documents, ["contribution", "novel", "propose"])
        
        result = chain.run(text=combined_text)
        contributions = [item.strip() for item in result.split('\n') if item.strip() and not item.strip().startswith('Key')]
        
        return contributions[:5]  # Limit to 5 contributions
    
    def _extract_methodology(self, documents: List[Document]) -> str:
        """Extract methodology from the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following research paper text, summarize the methodology or approach used in this study. 
            Focus on the methods, techniques, algorithms, or experimental setup:

            {text}

            Methodology:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Combine relevant chunks
        combined_text = self._combine_relevant_chunks(documents, ["method", "approach", "algorithm", "experiment"])
        
        result = chain.run(text=combined_text)
        return result.strip()
    
    def _extract_results(self, documents: List[Document]) -> str:
        """Extract results from the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following research paper text, summarize the key results, findings, or outcomes. 
            Focus on quantitative results, performance metrics, and main discoveries:

            {text}

            Results:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Combine relevant chunks
        combined_text = self._combine_relevant_chunks(documents, ["result", "finding", "performance", "evaluation"])
        
        result = chain.run(text=combined_text)
        return result.strip()
    
    def _extract_limitations(self, documents: List[Document]) -> str:
        """Extract limitations from the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following research paper text, identify and summarize the limitations, constraints, or shortcomings mentioned by the authors:

            {text}

            Limitations:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Combine relevant chunks
        combined_text = self._combine_relevant_chunks(documents, ["limitation", "constraint", "shortcoming", "weakness"])
        
        result = chain.run(text=combined_text)
        return result.strip() if result.strip() else "No specific limitations mentioned in the paper."
    
    def _extract_future_work(self, documents: List[Document]) -> str:
        """Extract future work from the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following research paper text, identify and summarize any future work, next steps, or research directions mentioned:

            {text}

            Future Work:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Combine relevant chunks
        combined_text = self._combine_relevant_chunks(documents, ["future", "next", "extension", "direction"])
        
        result = chain.run(text=combined_text)
        return result.strip() if result.strip() else "No specific future work mentioned in the paper."
    
    def _generate_overall_summary(self, documents: List[Document]) -> str:
        """Generate an overall summary of the paper"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Provide a comprehensive but concise summary of this research paper in 3-4 sentences. 
            Cover the main problem, approach, key findings, and significance:

            {text}

            Overall Summary:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Use summarize chain for better handling of long documents
        summarize_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        
        result = summarize_chain.run(documents)
        return result.strip()
    
    def _extract_technical_details(self, documents: List[Document]) -> Dict[str, str]:
        """Extract technical details like datasets, tools, etc."""
        details = {}
        
        # Extract datasets
        dataset_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following text, identify any datasets, benchmarks, or data sources mentioned:

            {text}

            Datasets:
            """
        )
        
        tools_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following text, identify any tools, frameworks, libraries, or software mentioned:

            {text}

            Tools:
            """
        )
        
        try:
            # Extract datasets
            dataset_chain = LLMChain(llm=self.llm, prompt=dataset_prompt)
            dataset_text = self._combine_relevant_chunks(documents, ["dataset", "benchmark", "data"])
            details["datasets"] = dataset_chain.run(text=dataset_text).strip()
            
            # Extract tools
            tools_chain = LLMChain(llm=self.llm, prompt=tools_prompt)
            tools_text = self._combine_relevant_chunks(documents, ["tool", "framework", "library", "software"])
            details["tools"] = tools_chain.run(text=tools_text).strip()
            
        except Exception as e:
            logger.warning(f"Error extracting technical details: {e}")
            details = {"datasets": "Not specified", "tools": "Not specified"}
        
        return details
    
    def _combine_relevant_chunks(self, documents: List[Document], keywords: List[str]) -> str:
        """Combine chunks that contain relevant keywords"""
        relevant_chunks = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in keywords):
                relevant_chunks.append(doc.page_content)
        
        # If no relevant chunks found, use first few chunks
        if not relevant_chunks:
            relevant_chunks = [doc.page_content for doc in documents[:3]]
        
        # Combine and limit length
        combined = ' '.join(relevant_chunks)
        return combined[:8000]  # Limit to avoid token limits