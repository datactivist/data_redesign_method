# Intuitiveness

[![DOI](https://zenodo.org/badge/685140191.svg)](https://zenodo.org/badge/latestdoi/685140191)

<p align="center">
  <img src="images/gear_cube_logo.svg" alt="Intuitiveness Gear Cube" width="120"/>
</p>

<p align="center">
  <strong>Intuitiveness as the next stage of open data</strong><br>
  <em>Dataset design and complexity</em>
</p>

---

## What is Intuitiveness?

**Intuitiveness** is a methodology and Python package for transforming raw, complex datasets into purpose-built data that directly answers your questions.

The method works through a **descent-ascent cycle**:
- **Descent** (L4 → L0): Strip away complexity to find the core truth
- **Ascent** (L0 → L3): Rebuild with YOUR intent, adding only relevant dimensions

### The 5 Levels of Abstraction

| Level | Name | Description |
|-------|------|-------------|
| **L4** | Raw Dataset | Original tabular data |
| **L3** | Entity Graph | Knowledge graph of relationships |
| **L2** | Domain Categories | Grouped by semantic domains |
| **L1** | Feature Vector | Unified numeric representation |
| **L0** | Core Datum | Single atomic value (the truth) |

## Features

- **Natural Language Search**: Query French open data (data.gouv.fr) in plain French using SmolLM3-3B
- **Interactive Streamlit App**: Visual descent-ascent workflow
- **Knowledge Graph**: Neo4j-powered entity relationships
- **Semantic Matching**: AI-assisted domain categorization

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ArthurSrz/intuitiveness.git
cd intuitiveness

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Scientific Article

This work is documented in a peer-reviewed research paper:

- **PDF**: [scientific_article/Intuitiveness.pdf](scientific_article/Intuitiveness.pdf)
- **Revised (v2)**: [scientific_article/v2_intuitive_datasets_revised.md](scientific_article/v2_intuitive_datasets_revised.md)

## Prerequisites

### Neo4j Database
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' neo4j:latest
```

### HuggingFace Token (for NL queries)
Set `HF_TOKEN` environment variable or add to `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_token_here"
```

## Project Structure

```
intuitiveness/
├── intuitiveness/          # Core package
│   ├── data_sources/       # MCP client & NL query engine
│   ├── services/           # data.gouv.fr search service
│   └── ui/                 # Streamlit components
├── scientific_article/     # Research paper
├── app.py                  # Main Streamlit app
└── requirements.txt
```

## Acknowledgments

Part of the [Dataflow](https://dataflow.hypotheses.org/) research project, funded by:
- **Datactivist**
- **UNESCO Chair in AI and Data Science for Society**

---

<p align="center">
  <sub>Designed by Arthur Sarazin & Mathis Mourey</sub>
</p>
