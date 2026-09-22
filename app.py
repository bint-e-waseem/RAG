import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

st.title("🎓 GCU University Admission Assistant")


@st.cache_resource(show_spinner="Loading GCU data...")
def build_rag_chain():
    urls = [
        'https://gcu.edu.pk/',
        'https://gcu.edu.pk/administration.php',
        'https://gcu.edu.pk/fee-structure.php',
        'https://gcu.edu.pk/financial-aid.php',
    ]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # Use a free local embedding model (no OpenAI key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    system_prompt = (
        """
You are GCU University Admission Assistant.

Your job is to help students with questions about
GCU University using only the information provided
in the retrieved context.

You can answer questions about:
- admission requirements
- degree programs
- eligibility
- application process
- fees
- scholarships
- departments
- important dates
- university contact information

Rules:
1. Use only the retrieved GCU information.
2. Do not invent or guess information.
3. If the answer is not available in the retrieved
   context, clearly say that you could not find
   the information in the available GCU resources.
4. Keep answers clear and student-friendly.
5. When possible, mention the source page.

Retrieved context:
{context}
"""
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


rag_chain = build_rag_chain()

query = st.chat_input("Ask about GCU admissions, programs, fees, eligibility...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching GCU resources..."):
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
