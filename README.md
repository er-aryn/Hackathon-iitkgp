# README.md

---

## Backstory Consistency Checker: Global Narrative Reasoning

This project provides a modular reasoning pipeline designed for the Kharagpur Data Science Hackathon 2026. The system evaluates whether a character's hypothetical backstory is logically and causally compatible with a long-form novel of over 100,000 words.

---

### Overview

Large Language Models typically struggle with global consistency in long narratives where meaning depends on how constraints and causal pathways accumulate over time. This solution addresses that challenge by transforming the problem into a structured classification task requiring careful evidence aggregation and causal reasoning.

---

### Key Features

* **Long-Context Management**: Ingests full-length novels using a chunking strategy of 800 words per segment to maintain semantic coherence.
* **Constraint Tracking**: Identifies specific claims within a backstory and tests them against the narrative's established rules and character developments.
* **Evidence-Based Decisions**: Aggregates signals from multiple parts of the text to support or challenge a backstory claim.
* **Binary Classification**: Produces a judgment of 1 (Consistent) or 0 (Contradict) along with a detailed evidence rationale.

---

### Technical Stack

* **Framework**: Pathway (Python framework for enterprise-grade data processing).


* **LLM Orchestration**: Ollama (running qwen2.5:3b).
* **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2).
* **Vector Store**: Custom implementation for indexing and searching long documents.

---

### Project Structure

* **Data/src/ingestion.py**: Handles novel loading, chunking, and table creation using Pathway.
* **Data/src/retrieval.py**: Implements the semantic search engine and vector store persistence.
* **Data/src/llm_judge.py**: Extracts claims from backstories and uses an LLM to judge consistency against retrieved passages.
* **Data/src/pipeline.py**: The main execution script that runs the end-to-end consistency check.
* **Data/src/baseline.py**: Provides a TF-IDF and Logistic Regression baseline for performance comparison.

---

### Installation and Setup

1. Install required dependencies:
```bash
pip install pathway pandas sentence-transformers ollama scikit-learn tqdm

```


2. Ensure Ollama is installed and the model is pulled:
```bash
ollama pull qwen2.5:3b

```


3. Place novels in the `Data/Novels` directory and input data in `Data/test.csv`.

---

### Usage

To run the full pipeline on the test dataset:

```bash
python Data/src/pipeline.py

```

To validate the model's accuracy against the training data:

```bash
python Data/src/pipeline.py validate

```

---

### Methodology

#### 1. Ingestion

The system utilizes Pathway to ingest raw text novels. Each novel is broken into 800-word chunks, which are then indexed to ensure the system can handle documents exceeding the standard context window of modern LLMs.

---

#### 2. Claim-Based Retrieval

Instead of searching for the entire backstory at once, the system extracts high-level claims. It then performs semantic searches to find the most relevant evidence passages within the novel that either support or refute these specific claims.

---

#### 3. Causal Judgment

The LLM Judge receives the backstory and the aggregated evidence passages. It determines if the character's early-life events, beliefs, and ambitions are compatible with the world rules and character development observed in the novel.

---

### Deliverables

* **results.csv**: A CSV file containing Story ID, Prediction (0 or 1), and a brief rationale.


* **Technical Report**: A comprehensive document detailing the handling of long context and causal signaling.


* **Reproducible Code**: Modular Python scripts for end-to-end execution.


---

### Team: Datafor | Event: Kharagpur Data Science Hackathon 2026
  
