#!/usr/bin/env python3
"""
Setup script for PaperSnap - AI-powered research paper summarization tool
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="papersnap",
    version="1.0.0",
    author="PaperSnap Team",
    author_email="contact@papersnap.dev",
    description="AI-powered tool for generating clean, structured summaries of research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/papersnap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "papersnap=papersnap.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "papersnap": ["*.py"],
    },
)