# Data redesign method

[![DOI](https://zenodo.org/badge/685140191.svg)](https://zenodo.org/badge/latestdoi/685140191)

![](https://dataflow.hypotheses.org/files/2023/05/cropped-dataflow-high-resolution-color-logo-1.png)

This repo contains [the scientific article](https://github.com/datactivist/data_redesign_method/blob/main/intuitive_datasets_short_ffinal.pdf) (peer review in progress) and the [pseudo code](https://github.com/datactivist/data_redesign_method/tree/main/features) of the data redesign method, designed by @Arthur Sarazin & Mathis Mourey.

It is one of the first results of the research project [Dataflow](https://dataflow.hypotheses.org/) funded by Datactivist and the UNESCO Chair in AI and Data science for Society.

## 📄 Scientific Article

- **Original**: [intuitive_datasets_short_ffinal.pdf](intuitive_datasets_short_ffinal.pdf)
- **Revised (v2)**: [scientific_article/intuitive_datasets_revised_v2.md](scientific_article/intuitive_datasets_revised_v2.md) - Includes the Geopost case study.

## 📦 Python Package & Implementation

This package implements the Data Redesign Method, allowing for the structured reduction and ascent of dataset complexity.

### Structure

- `intuitiveness/`: Core package containing complexity levels and redesign logic.
- `app.py`: Interactive Streamlit application for the Q&A workflow.
- `examples/`: Demonstration scripts.

### Prerequisites

The interactive Streamlit app requires the following external services:

#### 1. Neo4j Database

A Neo4j instance is required for storing and querying the knowledge graph (L3).

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest
```

Or use [Neo4j Desktop](https://neo4j.com/download/) or [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier available).

#### 2. Ollama with a Small LLM

[Ollama](https://ollama.ai/) is used for entity discovery and data model generation.

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the SmolLM2 model (recommended - small and fast)
ollama pull hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M

# Alternative: use a smaller model
ollama pull qwen2:0.5b
```

#### 3. Embedding Model (automatic)

The app uses `sentence-transformers` with the `intfloat/multilingual-e5-small` model for semantic matching during domain categorization (L3 -> L2) and for L4 -> L3 knowledge graph creation. This is downloaded automatically on first use.

### Installation

```bash
# Clone the repository
git clone https://github.com/datactivist/data_redesign_method.git
cd data_redesign_method

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `pandas`, `networkx`, `matplotlib`, `scipy` - Core data processing
- `sentence-transformers` - Semantic similarity for domain matching
- `streamlit` - Interactive web application

### Running the Streamlit App

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the app
streamlit run app.py
```

The app will guide you through the descent-ascent cycle:
1. **L4 -> L3**: Upload raw data, define entities with AI assistance, build knowledge graph
2. **L3 -> L2**: Categorize by domains using semantic matching
3. **L2 -> L1**: Extract feature vectors
4. **L1 -> L0**: Compute atomic metrics

## 🚀 Demonstration

### Quarto Notebook Demo

For a step-by-step visual walkthrough of the method, check out the Quarto notebook:

- **File**: [demo.qmd](demo.qmd)
- **Description**: A literate programming document that explains each step of the reduction process (Level 4 -> Level 0) with code and narrative.

To view or render the notebook, ensure you have [Quarto](https://quarto.org/) installed.

### Running the Geopost Demo Script

The demo script `examples/geopost_demo.py` demonstrates the full pipeline using Geopost data:

1.  **Level 4**: Loads raw CSV data.
2.  **Level 3**: Builds a Knowledge Graph connecting Indicators to Departments and Sources.
3.  **Level 2**: Extracts a table of indicators for the 'Finance' department.
4.  **Level 1**: Extracts the list of indicator names.
5.  **Level 0**: Calculates the total count of Finance indicators.

To run the demo:

```bash
python examples/geopost_demo.py
```
