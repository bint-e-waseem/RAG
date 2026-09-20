# RAG
# 🎓 GCU University Admission Assistant

A Streamlit-based **Retrieval-Augmented Generation (RAG)** chatbot that answers student queries about **Government College University (GCU)** admissions, programs, fees, scholarships, and more — powered by **Google Gemini** and a local vector database.

---

## 📖 Overview

The **GCU University Admission Assistant** is an AI-powered conversational agent that scrapes official GCU web pages, converts them into embeddings, stores them in a Chroma vector store, and uses **Google Gemini 2.5 Flash** to generate accurate, context-grounded answers.

It ensures responses are based **only** on official GCU data — no hallucinations, no guesswork.

---

## ✨ Features

- 🔍 **Automated Web Scraping** of official GCU pages using `UnstructuredURLLoader`
- 🧠 **Local Embeddings** via `sentence-transformers/all-MiniLM-L6-v2` (no OpenAI key required)
- 🗂️ **Chroma Vector Store** for fast semantic similarity search
- 🤖 **Google Gemini 2.5 Flash** as the LLM for natural, student-friendly responses
- 🔗 **LangChain RAG Pipeline** (`create_retrieval_chain` + `create_stuff_documents_chain`)
- 💬 **Streamlit Chat UI** with conversational memory-style interaction
- ⚡ **Cached Resource Loading** for faster subsequent queries
- 🛡️ **Strict Grounding** — refuses to answer outside the retrieved context

---

## 🏗️ Architecture

```
                    ┌────────────────────────┐
                    │   GCU Official URLs    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ UnstructuredURLLoader  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Recursive Text Splitter│
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ HuggingFace Embeddings │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Chroma Vector Store  │
                    └───────────┬────────────┘
                                │
        User Query ──► Retriever ──► Gemini 2.5 Flash ──► Answer
```

---

## 🧠 How Google Gemini Is Used

### 1. **Model Initialization**
```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
```

- **`gemini-2.5-flash`** — Google's fast, cost-efficient, multimodal LLM from the Gemini 2.5 family.
- **`temperature=0.2`** — Keeps responses **deterministic and factual**, which is critical for an admission assistant where accuracy matters more than creativity.

### 2. **Role in the RAG Pipeline**

Gemini is the **final reasoning engine** of the pipeline. Here's the exact flow:

| Step | Component | Role |
|------|-----------|------|
| 1 | `UnstructuredURLLoader` | Loads raw GCU web content |
| 2 | `RecursiveCharacterTextSplitter` | Splits into 1000-char chunks (100 overlap) |
| 3 | `HuggingFaceEmbeddings` | Converts chunks → vectors |
| 4 | `Chroma` | Stores vectors for similarity search |
| 5 | `Retriever` | Fetches top-6 relevant chunks for a query |
| 6 | **`ChatGoogleGenerativeAI` (Gemini)** | **Generates the final answer from retrieved context** |
| 7 | `create_stuff_documents_chain` | "Stuffs" retrieved docs into the prompt |
| 8 | `create_retrieval_chain` | Ties retriever + LLM together |

### 3. **Prompt Engineering with Gemini**

Gemini receives a carefully crafted **system prompt** that enforces:

- ✅ Use **only** retrieved GCU context
- ✅ Never invent or hallucinate information
- ✅ Politely refuse if the answer isn't in context
- ✅ Keep answers student-friendly
- ✅ Cite the source page when possible

The `{context}` placeholder is dynamically filled with the retrieved documents, and `{input}` is the user's question.

### 4. **Why Gemini 2.5 Flash?**

| Benefit | Explanation |
|---------|-------------|
| ⚡ **Fast** | Flash variant = low latency, great for chat |
| 💰 **Cost-effective** | Cheaper than Pro models |
| 🎯 **Accurate** | Strong instruction-following for RAG |
| 🌐 **Multimodal-ready** | Can be extended to images/PDFs later |

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- A **Google API Key** for Gemini ([Get one here](https://aistudio.google.com/app/apikey))

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gcu-admission-assistant.git
cd gcu-admission-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install streamlit langchain langchain-google-genai langchain-community \
            chromadb sentence-transformers unstructured python-dotenv
```

### 4. Configure environment variables
Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
gcu-admission-assistant/
│
├── app.py                  # Main Streamlit application
├── .env                    # API keys (not committed)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── chroma_db/              # Auto-generated vector store
```

---

## 💬 Usage

Once the app is running, open your browser at `http://localhost:8501` and ask questions like:

- *"What are the admission requirements for BS Computer Science?"*
- *"How much is the fee for intermediate programs?"*
- *"Does GCU offer financial aid?"*
- *"What departments are available at GCU?"*
- *"What is the contact information of GCU?"*

The assistant will retrieve relevant info from GCU's official pages and respond using **Gemini**.

---

## 🔧 Configuration

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| `model` | `ChatGoogleGenerativeAI` | `gemini-2.5-flash` | Gemini model version |
| `temperature` | `ChatGoogleGenerativeAI` | `0.2` | Response randomness |
| `chunk_size` | `RecursiveCharacterTextSplitter` | `1000` | Text chunk length |
| `chunk_overlap` | `RecursiveCharacterTextSplitter` | `100` | Overlap between chunks |
| `k` | `retriever` | `6` | Number of docs retrieved |
| `search_type` | `retriever` | `similarity` | Retrieval strategy |

---

## 🛡️ Guardrails

The assistant is designed to **never hallucinate**. If the answer isn't in the retrieved GCU context, it will reply:

> *"I could not find this information in the available GCU resources."*

This is enforced through the system prompt passed to Gemini.

---

## 🧪 Example Interaction

**User:** *What is the fee structure for BS programs?*

**Assistant:**
> Based on the GCU fee structure page, the fee for BS programs varies by department. For example, BS Computer Science has a semester fee of PKR XX,XXX, while BS Physics is PKR XX,XXX. Please refer to https://gcu.edu.pk/fee-structure.php for the latest breakdown.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM | **Google Gemini 2.5 Flash** |
| Framework | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | Chroma |
| Loader | UnstructuredURLLoader |
| Config | python-dotenv |

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [Google AI Studio](https://aistudio.google.com/) for Gemini API
- [LangChain](https://www.langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the UI
- [GCU Official Website](https://gcu.edu.pk/) for source data

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with ❤️ for GCU students.**
