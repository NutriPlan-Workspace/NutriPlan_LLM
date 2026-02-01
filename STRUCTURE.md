# Repository Structure

This repository has been reorganized to focus on LLM Finetuning and RAG Testing.

## `finetune/`
Contains code and data related to finetuning the LLM.
- `dataset/`: Contains training data files (e.g., `train.jsonl`).
- `finetune_llm.ipynb`: Jupyter notebook for the finetuning process.

## `rag/`
Contains code and tests for the RAG (Retrieval-Augmented Generation) pipeline.
- `chunking_and_db.ipynb`: Notebook for chunking data and inserting it into the MongoDB database.
- `generate_vector_embeddings.py`: Script to recompute and update food embeddings in MongoDB.
- `test_rag_pipeline.ipynb`: Playground notebook for testing RAG components and OpenAI/LLM interactions.
- `test_web_search.py`: Script to test the Web Search tool.
- `config.py`: Configuration file for environment variables and settings (MongoDB, Models).
- `constants.py`: definitions of constants (e.g., `CATEGORY_LABELS`).

## `misc/`
Contains other notebooks or legacy files.
- `nutriplan.ipynb`: The original main notebook (content preserved).

## Root
- `.gitignore`
- `STRUCTURE.md` (This file)

## Usage
- To run RAG scripts, ensure you are in the `rag/` directory or have the repository root in your `PYTHONPATH` to resolve imports from `config.py` and `constants.py`.
- Finetuning can be run directly from `finetune/finetune_llm.ipynb`.
