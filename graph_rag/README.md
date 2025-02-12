# Graph RAG 🌟

A beginner-friendly graph-based Retrieval Augmented Generation (RAG) system designed for insurance policy analysis. This system combines the power of graph databases with AI to help understand and analyze insurance policies.

[Hosted retriever notebook](https://colab.research.google.com/drive/1QrPbC3a9dlSjNsMoWkSl4cRuPGjopVZ2?usp=sharing)

## What is RAG? 🤔

RAG (Retrieval Augmented Generation) is a technique that helps AI models give more accurate answers by:
1. First finding relevant information from a database
2. Then using that information to generate accurate responses

Think of it like an AI assistant that first looks up information in a reference book before answering your question!

## How This System Works 🔄

Our system works in two main phases:

### 1. Learning Phase (Ingestion) 📚
- Takes insurance policy documents (in markdown format)
- Breaks them into smaller, manageable pieces
- Uses AI (GPT-4) to understand and extract important concepts
- Creates connections between related concepts
- Stores everything in a graph database (like a smart mind map!)

### 2. Question-Answering Phase (Retrieval) 💡
- Takes your question about insurance coverage
- Finds the most relevant parts of the policy
- Follows connections to related information
- Uses AI to analyze your question against the policy
- Gives you a clear, structured answer

## Getting Started 🚀

### 1. Set Up Your Environment 🛠️

First, make sure you have Python 3.9+ installed. Then:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies using uv
uv pip install -r requirements.txt
```

### 2. Configuration ⚙️
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your settings:
   ```env
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4-turbo-preview
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   ```

### 3. Start the Database 🗄️
```bash
docker-compose --env-file .env -f memgraph-platform/docker-compose.yml up
```

## Using the System 🎯

Our system provides three main commands:

### 1. Load Policy Documents 📝
```bash
python -m graph_rag.cli ingest --verbose
```
This command:
- Reads the policy document from `data/llamaparse/motor_2021.md`
- Processes it and stores it in the database
- Creates special indexes for quick searching

### 2. Ask Questions ❓
```bash
python -m graph_rag.cli query --query "Your question here"

Example:
```bash
python -m graph_rag.cli query --query "If I hit a deer and damaged my front bumper, am I covered?"
```

### 3. Process Multiple Questions 📊
```bash
python -m graph_rag.cli batch --claims_path PATH_TO_CLAIMS_FILE --verbose
```
This is useful to run the benchmark.

Need help? Try:
```bash
python -m graph_rag.cli --help
python -m graph_rag.cli <command> --help
```

## How Answers Are Formatted 📋

The system gives answers in a clear, structured format:
```json
{
    "covered": true/false,
    "explanation": "Clear explanation of why",
    "limits": ["Any coverage limits that apply"],
    "exclusions": ["Any relevant exclusions"]
}
```

## System Components 🏗️

### Core Parts
- `main.py`: Handles document processing
- `retrieve.py`: Manages question answering
- `retriever/`: Core search logic
- `database/`: Database operations
- `services/`: AI integration
- `utils/`: Helper functions
- `config/`: System settings
- `experiments/`: Testing and evaluation

### Visual Overview
```
                                    ┌─────────────────┐
                                    │  Policy Docs    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Text Chunking  │
                                    └────────┬────────┘
                                             │
                     ┌───────────────────────┴──────────────────────┐
                     │                                              │
            ┌────────▼────────┐                            ┌────────▼────────┐
            │   KG Extraction │                            │    Embeddings   │
            │      (GPT-4)    │                            │  Generation     │
            └────────┬────────┘                            └────────┬────────┘
                     │                                              │
                     │                                              │
            ┌────────▼───────────────────────────────────────┐      │
            │                   Memgraph                     │◄─────┘
            │    (Nodes, Relationships, Vector Index)        │
            └───────────────────┬────────────────────────────┘
                                │
                                │         ┌─────────────┐
                                │         │   Query     │
                                │         └──────┬──────┘
                                │                │
                     ┌──────────▼────────────────▼──────────┐
                     │        Retrieval Pipeline            │
                     │  1. Vector Similarity Search         │
                     │  2. Graph Relationship Traversal     │
                     │  3. Context Building                 │
                     │  4. GPT Analysis                     │
                     └─────────────────┬────────────────────┘
                                       │
                                ┌──────▼──────┐
                                │   Answer    │
                                └─────────────┘
```

## Important Notes 📌

- This is a proof of concept - great for learning but not for production use
- Requires an OpenAI API key
- Works best with insurance policy documents
- Performance depends on your database settings


## Caching 💾

To work faster (and cheaper), the system remembers:
- Document embeddings
- AI responses
- Knowledge graph data

Cache location: `cache/` directory

## Need Help? 🆘

- Check the error messages - they're designed to be helpful
- Look at the examples in the code
- Make sure your API keys are set correctly
- Verify your database is running

Remember: This is a learning tool - feel free to experiment and learn from how it works! 🌟