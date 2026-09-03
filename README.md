# RAG
# RAG Demo: Victoria on Move

A production-ready Retrieval-Augmented Generation (RAG) application that demonstrates the integration of multiple AI services for intelligent question-answering using website content.

## 📋 Overview

This project showcases a complete RAG pipeline that ingests website content, processes it, and enables natural language querying. The application leverages **Pinecone** vector database and **Alibaba Cloud Model Studio Qwen** to provide accurate, context-aware responses based on the Victoria on Move moving services website.

## 🚀 Features

- **Real-time Web Scraping**: Automatically extracts and processes content from live websites
- **Intelligent Chunking**: Uses recursive character text splitting with configurable chunk sizes
- **Vector Embeddings**: Utilizes `text-embedding-v4` for high-quality semantic representations
- **Vector Database**: Pinecone for scalable, efficient similarity search
- **LLM Integration**: Qwen LLM for generating coherent, context-aware responses
- **Interactive Interface**: Streamlit UI for real-time querying
- **Dual Implementation**: Both Jupyter Notebook and Streamlit app versions included

## 🏗️ Architecture

```
┌─────────────────┐
│   Website URLs   │
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Unstructured   │
│   URLLoader     │
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Text Splitter  │
│  (Chunking)     │
└────────┬─────────┘
         ▼
┌─────────────────┐
│   Embeddings    │
│ (text-embedding)│
└────────┬─────────┘
         ▼
┌─────────────────┐
│    Pinecone     │
│  Vector Store   │
└────────┬─────────┘
         ▼
┌─────────────────┐
│   Retriever     │
│  (Similarity)   │
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Qwen LLM       │
│  (Generation)   │
└────────┬─────────┘
         ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

## 📦 Technology Stack

- **LangChain**: Framework for LLM application development
- **Pinecone**: Vector database for similarity search
- **Alibaba Cloud Model Studio**: Qwen LLM and embedding models
- **Streamlit**: Interactive web interface
- **Python 3.10+**: Core programming language
- **Unstructured**: Document parsing and extraction

## 📁 Project Structure

```
rag-demo/
├── app.py                          # Streamlit application
├── rag_demo_pinecone_qwen.ipynb    # Jupyter notebook with full implementation
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- API Keys:
  - **Alibaba Cloud Model Studio API Key** (for Qwen LLM and embeddings)
  - **Pinecone API Key** (for vector database)
  - **Hugging Face API Key** (for embedding models)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-demo.git
   cd rag-demo
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```env
   # Alibaba Cloud Model Studio
   DASHSCOPE_API_KEY=your_dashscope_api_key_here
   DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
   
   # Pinecone
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=victoria-on-move-qwen-rag
   PINECONE_NAMESPACE=victoria-on-move
   
   # Hugging Face
   HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

## 🚀 Usage

### Streamlit Application

Run the interactive web interface:
```bash
streamlit run app.py
```

The application will:
1. Automatically load and process the Victoria on Move website
2. Create vector embeddings and store them in Pinecone
3. Provide a chat interface for asking questions

### Jupyter Notebook

For a detailed, step-by-step implementation:
```bash
jupyter notebook rag_demo_pinecone_qwen.ipynb
```

The notebook includes:
- Complete code walkthrough
- Interactive cells for experimentation
- Output visualization
- Testing with sample questions

## 💻 Sample Queries

The RAG system can answer various questions about Victoria on Move services:

| Query | Context |
|-------|---------|
| "What types of trucks do they offer?" | Returns fleet information with sizes and pricing |
| "What insurance do they provide?" | Details about transit and liability insurance |
| "What cities can they move to from Melbourne?" | Interstate moving destinations |
| "Do they provide packing services?" | Information about packing/unpacking services |

## 🧪 Technical Implementation Details

### 1. Document Processing
```python
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
```

### 2. Vector Database Setup
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="victoria-on-move"
)
```

### 3. RAG Chain
```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. "
    "If you don't know, say that you don't know."
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

## 📊 Performance Considerations

- **Chunk Size**: Optimized at 1000 tokens with 150 token overlap
- **Retrieval**: Top-6 most relevant documents for comprehensive context
- **Embedding Dimension**: 384-dimensional vectors for efficient storage
- **Model**: Qwen-plus for balanced performance and quality

## 🎯 Use Cases

- **Customer Support**: Answer questions about services, pricing, and policies
- **Content Analysis**: Quickly find information within large document collections
- **Knowledge Base**: Build a searchable repository of website content
- **Research**: Extract and query information from multiple sources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alibaba Cloud** for Model Studio and Qwen models
- **Pinecone** for vector database services
- **LangChain** for the LLM application framework
- **Streamlit** for the interactive UI framework

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Built with ❤️ using Alibaba Cloud Model Studio, Pinecone, and LangChain**
