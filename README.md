# NutriPlan LLM

This repository handles the Large Language Model (LLM) components of the NutriPlan application, specifically focusing on RAG (Retrieval-Augmented Generation) and model finetuning.

## Features

- **RAG Pipeline**:
  - **Food Database**: Chunking and vector search for food items, recipes, and nutritional info.
  - **User Manual**: Chunking and vector search for application usage guides.
  - **Web Search**: Integration for retrieving real-time information.
- **LLM Finetuning**: Tools and notebooks for finetuning models on custom datasets.
- **Frontend Commands**: Generation of JSON commands to control the NutriPlan UI.

## Directory Structure

```
llm/
├── rag/                        # RAG Pipeline and Testing
│   ├── food_chunking.ipynb     # Food database chunking & embedding
│   ├── user_manual_chunking.ipynb # User manual chunking & embedding
│   ├── rag_test.ipynb          # Main RAG pipeline testing notebook
│   └── config.py               # Configuration & Environment loading
│
├── finetune/                   # Model Finetuning
│   ├── dataset/                # Training datasets
│   └── finetune_llm.ipynb      # Finetuning notebook
│
├── misc/                       # Helper scripts and legacy files
│   ├── generate_vector_embeddings.py
│   ├── test_web_search.py
│   └── nutriplan.ipynb
│
├── .env                        # Environment variables (secrets)
└── .env.example                # Example environment file
```

## Setup

1.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    ```
    Required variables:
    - `MONGODB_URI`: Connection string for MongoDB Atlas.
    - `OPENAI_BASE_URL`: URL for the LLM inference server (e.g., vLLM or OpenAI).
    - `HF_TOKEN`: HuggingFace token for model access.
    - `GITHUB_TOKEN`: GitHub token (if using GitHub models).
    - `REDIS_URL`: URL for Redis (if applicable).

2.  **Dependencies**:
    Ensure you have the necessary Python packages installed.
    *Note: A `requirements.txt` should be generated if not present.*
    Common dependencies include: `langchain`, `pymongo`, `openai`, `python-dotenv`, `sentence-transformers`.

## Usage

### RAG Operations
- **Chunking Data**: Run `rag/food_chunking.ipynb` or `rag/user_manual_chunking.ipynb` to process data and populate the vector database (MongoDB).
- **Testing RAG**: Open `rag/rag_test.ipynb` to interact with the full pipeline, test queries, and verify tool usage.

### Finetuning
- Navigate to `finetune/finetune_llm.ipynb` to run the training process on your dataset.

### Configuration
- Modify `rag/config.py` to adjust global settings or loaded environment variables.
