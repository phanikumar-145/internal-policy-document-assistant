# Internal Policy Document Assistant (RAG)

A local, privacy-preserving web application that allows employees to upload company policy PDF documents and ask questions about their contents using Retrieval-Augmented Generation (RAG).

Answers are generated strictly from retrieved policy excerpts with PDF filename and page citations. If an answer is not present in the uploaded documents, the assistant responds:

> `I could not find this information in the available policy documents.`

## Tech Stack

- **Python** (3.10+)
- **Streamlit** (Web interface)
- **LangChain** (RAG orchestration)
- **PyPDF** (PDF text extraction)
- **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2` local embeddings)
- **FAISS** (Local vector search index)
- **Ollama** (`llama3.2:3b` local LLM)

## Prerequisites

1. **Python 3.10 or newer** installed on your system.
2. **Ollama must be installed separately** before running the assistant. Download and install Ollama for Windows (or macOS/Linux) from [https://ollama.com/download](https://ollama.com/download).

## Setup and Execution (Windows)

Run the following commands in Command Prompt or PowerShell from the project directory:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama serve
streamlit run app.py
```

*(Note: On Windows, if the Ollama desktop background service is already running, `ollama serve` may report that port `11434` is already active—in that case, proceed directly to `streamlit run app.py`.)*

## Usage

1. Open the local URL shown by Streamlit (typically `http://localhost:8501`).
2. Upload one or more company policy PDF files in the **Policy Documents** sidebar.
3. Click **Index Documents** and wait for the indexing status to confirm completion.
4. Type policy questions into the chat box to receive grounded answers and page-level source citations.
5. Use **Clear Chat** to reset conversation history or upload updated files and click **Index Documents** to re-index.

## Running Evaluation & Security Tests

With your virtual environment activated, run the held-out retrieval, grounding, citation, fallback, and prompt-injection test suite:

```
python evaluation.py
```


## Sample policy PDFs

Three fictional sample documents are included in the `documents/` folder:
- `leave_policy.pdf`
- `employee_handbook.pdf`
- `attendance_policy.pdf`

Start the Streamlit app, upload these PDFs using the app's PDF uploader, and click the indexing button. These are demonstration policies only; replace them with your organization's approved documents for real use.
