# Self-Healing RAG (Retrieval-Augmented Generation)

A resilient, agentic Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **LangChain**, **ChromaDB**, and **Groq (Llama 3.3 70B)**. 

Unlike traditional static RAG pipelines, this system features an interactive self-healing loop: if an answer is evaluated as ungrounded, incomplete, or off-topic relative to the retrieved context, the system autonomously rewrites the query and retries retrieval up to a designated threshold before delivering a response.

> [!NOTE]
> **Interface Notice**: This application currently runs as an interactive **Command Line Interface (CLI)** directly inside your terminal (`ragagent.py`). It does not currently include a graphical web or desktop user interface (GUI). Users wishing to add a web frontend can easily wrap the core LangGraph agent in a framework like Streamlit, Gradio, or FastAPI.

---

## 🌟 Architecture & Workflow

```
                   +------------------+
                   |  User Question   |
                   +--------+---------+
                            |
                            v
                   +------------------+
                   |     Retrieve     | <----------------+
                   |    (ChromaDB)    |                  |
                   +--------+---------+                  |
                            |                            |
                            v                            |
                   +------------------+                  |
                   |     Generate     |                  |
                   |  (Llama 3.3 70B) |                  |
                   +--------+---------+                  |
                            |                            |
                            v                            |
                   +------------------+                  |
                   |   Grade Answer   |                  |
                   |  (Evaluator LLM) |                  |
                   +--------+---------+                  |
                            |                            |
                            | [Conditional Route]        |
            +---------------+---------------+            |
            |                               |            |
     (Grade == PASS)                 (Grade == FAIL)     |
            |                               |            |
            v                               v            |
         +-----+                    +---------------+    |
         | END |                    |  Retry Limit  |    |
         +-----+                    |   Exceeded?   |    |
                                    +---+-------+---+    |
                                        |       |        |
                                     (No)      (Yes)     |
                                        |       |        |
                                        v       v        |
                            +------------------+  +----------+
                            | Rewrite Question |  | Give Up  |
                            +--------+---------+  +----+-----+
                                     |                 |
                                     +-----------------+---> END
```

### Self-Healing Cycle

1. **Document Ingestion (`ingest.py`)**: Loads raw text files, chunks them using `RecursiveCharacterTextSplitter`, and embeds them into a local vector database (`ChromaDB`).
2. **Retrieve (`retrieve`)**: Fetches top relevant context documents from ChromaDB based on the original or rewritten question.
3. **Generate (`generate`)**: Prompts the LLM (`llama-3.3-70b-versatile`) to generate a concise answer grounded in the retrieved documents.
4. **Grade (`grade_answer`)**: An evaluator node assesses whether the generated answer is strictly supported by the context without hallucination.
5. **Self-Correction (`rewrite_question`)**:
   - If the answer **fails** evaluation and retry count is below a **configurable max threshold** (default: `max_retries = 2` in `route_after_grading`), the question is autonomously rewritten with refined terminology to attempt better document matching, restarting the retrieval loop.
   - If retries are exhausted, the **`give_up`** node returns a transparent message indicating that reliable information could not be located in the knowledge base.

---

## 📊 Evaluation & Experiment Tracking

The system includes a reproducible evaluation harness and MLflow-based experiment tracking so that the self-healing behavior can be measured, compared across variants, and validated statistically.

### Variants
| Variant | Description |
| :--- | :--- |
| `single_pass` | Vanilla RAG — retrieve → generate → grade → end (no retry). |
| `one_retry` | Allows **1** autonomous query rewrite before fallback. |
| `two_retry` | Allows **2** autonomous query rewrites (the live `ragagent.py` default) before fallback. |

### Metrics (`eval/metrics.py`, all as an LLM judge on the same Groq model)
- **Hallucination rate** — fraction of answers judged to contain claims not supported by the retrieved context (lower is better).
- **Coverage rate** — fraction of answers judged to adequately address the question given the context (higher is better).
- **Citation rate** — fraction of answers containing an inline `[Source: ...]` citation.
- **Pass rate** — fraction of answers that pass the evaluator node's grounding check.

### Experiment Tracking (`eval/mlflow_tracking.py`)
Every evaluation run logs to MLflow (SQLite backend `mlflow.db` + local artifact store `mlartifacts/`):
- **Parameters** — variant, max_retries, model, temperature, chunk_size, top_k.
- **Metrics** — hallucination_rate, coverage_rate, citation_rate, pass_rate, avg_retries.
- **Artifacts** — per-question JSON results and a `predictions.jsonl` for downstream analysis.
- **Comparison report** — an MLflow run comparison table (`variant_comparison.csv`).

Statistical significance is computed per metric pair (`eval/stats.py`) using a **McNemar exact test** on matched per-question outcomes plus a **bootstrap confidence interval** for the rate delta, with a verdict of `SIGNIFICANT` only when p < 0.05 *and* the CI excludes zero.

### Running the Evaluation
```bash
# All variants on the full dataset
uv run python -m eval.evaluate --dataset eval/dataset.jsonl --experiment self-heal-rag

# A specific variant pair with a limited slice (avoids token/rate limits)
uv run python -m eval.evaluate --dataset eval/dataset.jsonl \
    --experiment self-heal-rag --variants single_pass one_retry --limit 15
```

### Measured Results (sample run)
> ⚠️ **Honest caveat:** the numbers below come from a *small* sample. On a 15-question set the two variants were statistically **indistinguishable** (McNemar p = 1.0, CI crosses zero): retry logic only helps when there are genuinely recoverable retrieval failures. On a 5-question set of **ambiguous** questions, coverage rose from **80% → 100% (+20pp)** with retries, but with n=5 this is **not statistically significant** (p = 1.0). Larger sample sizes are required before the improvement can be claimed with confidence. The observed trends are **suggestive, not conclusive**.

| Setting | Variant | Hallucination | Coverage | Citation | Pass |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 15 general questions | `single_pass` | 53.3% | 80.0% | 93.3% | 93.3% |
| 15 general questions | `one_retry` | 53.3% | 80.0% | 93.3% | 93.3% |
| 5 ambiguous questions | `single_pass` | 40.0% | 80.0% | 100% | 100% |
| 5 ambiguous questions | `one_retry` | 40.0% | **100.0%** | 100% | 100% |

**Interpretation:** On ambiguous queries the retry loop improved coverage by +20pp in this small run, but the effect did not reach statistical significance and did not reduce hallucination. The inference is limited by small sample size and LLM judge variability; a larger ambiguous dataset is needed to substantiate before claiming a fixed percentage improvement.

---

## 📁 Directory Structure

```text
self-heal-rag/
├── docs/                    # Input documents directory (.txt files)
│   ├── machine_learning.txt
│   └── python_basics.txt
├── chroma_db/              # Persistent ChromaDB vector database
├── ingest.py               # Document loading, splitting & vector DB ingestion
├── ragagent.py             # LangGraph state graph definition & CLI interface
├── main.py                 # Entry point script
├── eval/
│   ├── dataset.jsonl       # Labeled evaluation questions (factual / ambiguous / out-of-domain)
│   ├── evaluate.py         # CLI runner: `python -m eval.evaluate`
│   ├── baselines.py        # RAG variants: single_pass, one_retry, two_retry
│   ├── metrics.py          # Hallucination & coverage judges (LLM-as-judge) + citation checker
│   ├── mlflow_tracking.py  # Experiment logging (parameters, metrics, artifacts)
│   └── stats.py            # McNemar test + bootstrap CIs for paired significance
├── .env                    # Environment variables (API keys)
├── pyproject.toml          # Project dependencies and configuration
└── README.md               # Project documentation
```

> `mlflow.db` (SQLite) and `mlartifacts/` are generated by the evaluation harness and are git-ignored.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `>= 3.11`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- **Groq API Key**: Obtain from [Groq Console](https://console.groq.com/)

### Installation & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd self-heal-rag
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Install Dependencies**:
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using standard `pip`:
   ```bash
   pip install -e .
   ```

---

## 💻 Usage (Terminal / CLI)

> [!IMPORTANT]
> The system operates via **Terminal/CLI** input. There is no web GUI included by default.

### 1. Ingest Documents into Knowledge Base

Place your source text files inside the `docs/` directory, then run the ingestion script in your terminal:

```bash
python ingest.py
```

This will process all `.txt` files in `docs/`, split them into chunks of 300 characters (with 50-character overlap), and store them into ChromaDB under `./chroma_db`.

### 2. Run the Self-Healing RAG Agent

Launch the interactive terminal interface:

```bash
python ragagent.py
```

**Example Terminal Interaction**:

```text
Self-Healing RAG System
========================================
Type your question and press Enter.
Type 'quit' to exit.

Your question: What techniques prevent neural network overfitting?

Retrieving documents for question: What techniques prevent neural network overfitting?
Retrieved 3 documents from ChromaDB.
Generating answer for question: What techniques prevent neural network overfitting?
[GENERATE] Generated answer: len(112) characters

[GRADE] evaluating answer quality...
[GRADE] Result: PASS

[FINAL ANSWER] Regularization techniques like dropout and L2 penalty help prevent overfitting in neural networks.
```

---

## 🍴 Forking & Adapting for Your Custom Purpose

This repository is designed to serve as a modular, extensible boilerplate for building domain-specific self-healing RAG applications. Here is a step-by-step guide on how to fork, customize, and adapt it for your own project:

### 1. Fork & Clone

1. Click the **Fork** button at the top right of this GitHub repository page.
2. Clone your personal fork locally:
   ```bash
   git clone https://github.com/rishik0612/self-heal-rag.git
   cd self-heal-rag
   ```

### 2. Add Your Own Data & Custom Loaders (`ingest.py`)

- **Replace sample files**: Remove the default files in `docs/` and add your own `.txt` files (e.g. company FAQs, internal wikis, product manuals, medical notes).
- **Support PDFs, Markdown, or HTML**: Swap out `TextLoader` in `ingest.py` with other LangChain loaders:
  ```python
  from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

  # Example for PDF ingestion:
  loader = PyPDFLoader("path/to/document.pdf")
  documents = loader.load()
  ```
- **Tune Chunking Parameters**: Modify `chunk_size` and `chunk_overlap` based on your document structure:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,    # Increase for longer context blocks
      chunk_overlap=100, # Maintain semantic connection across chunks
      length_function=len
  )
  ```
- **Re-ingest Vector Embeddings**:
  ```bash
  python ingest.py
  ```

### 3. Customize LLMs & Embedding Models (`ragagent.py`)

- **Change Groq Model**: Update `model` parameter in `ragagent.py`:
  ```python
  llm = ChatGroq(
      model="llama-3.1-8b-instant", # Or "mixtral-8x7b-32768"
      temperature=0.2,
      api_key=os.environ.get("GROQ_API_KEY")
  )
  ```
- **Switch Provider (e.g. OpenAI, Anthropic, Ollama)**:
  Replace `ChatGroq` with your preferred provider:
  ```python
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
  ```

### 4. Fine-Tune Self-Healing Behavior (`ragagent.py`)

- **Adjust Max Retries**: In `route_after_grading`, change `max_retries`:
  ```python
  max_retries = 3 # Allow up to 3 query rewrite attempts
  ```
- **Customize Grading Rules**: Modify `grading_prompt` in `grade_answer` to enforce specific domain standards, compliance checks, or strict citation matching.
- **Add Web Search Fallback**: Update the `give_up` or `rewrite` node to query a web search engine (e.g., Tavily, DuckDuckGo) if internal documents lack the answer.

### 5. Adding a Graphical Interface (Web UI)

Since this project currently runs in the terminal, you can easily add a web interface by building on top of `ragagent.py`:

- **Streamlit**: Create a `app.py` file using Streamlit chat components (`st.chat_input` and `st.chat_message`) calling `rag_graph.invoke()`.
- **FastAPI / REST API**: Expose an endpoint (`POST /ask`) to receive questions from any frontend framework (React, Vue, Next.js).
- **Gradio**: Build a fast prototype UI using `gradio.ChatInterface`.

---

## 🛠️ Key Dependencies

| Dependency | Purpose |
| :--- | :--- |
| **`langgraph`** | Orchestrates agentic RAG workflows, state management, and conditional routing |
| **`langchain-groq`** | Interfacing with Groq's high-speed inference engine (`llama-3.3-70b-versatile`) |
| **`chromadb`** | Vector database for persistent document retrieval |
| **`langchain-text-splitters`** | Document chunking with `RecursiveCharacterTextSplitter` |
| **`python-dotenv`** | Managing environment configuration |
| **`mlflow`** | Reproducible experiment tracking (parameters, metrics, artifacts, run comparison) |
| **`pandas`** | Aggregate metrics and variant-comparison reporting |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
