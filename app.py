import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def main():
    st.title('RAG PDF Chatbot')


    # Sidebar file uploader
    st.sidebar.header("Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Check if files were uploaded
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return

    # Process uploaded PDFs
    all_documents = []
    st.sidebar.write('Processing uploaded PDF files...')

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = os.path.join("temp_files", uploaded_file.name)
        os.makedirs("temp_files", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF content using PyPDFLoader
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        all_documents.extend(documents)

    st.sidebar.write(f"Loaded {len(all_documents)} documents.")
    
    # Load environment variables
    load_dotenv()

    # Embedding and creating vector store
    st.sidebar.write("Embedding and creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings) 
    vectorstore = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory="path_to_local_directory")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Configure the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

    # Create the system and user prompt templates
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # User input
    st.header("Ask a Question")
    user_input = st.text_input("Enter your question:")

    if user_input:
        with st.spinner("Retrieving and generating response..."):
            response = rag_chain.invoke({"input": user_input})
            st.subheader("Answer")
            st.write(response["answer"])
            
    #Clear Cache
    
    if st.sidebar.button("Clear Cache & Restart"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()       

def cleanup_temp_files(temp_files):
    """Delete temporary files after processing."""
    for file_path in temp_files:
        try:
            os.remove(file_path)  # Delete the file
        except Exception as e:
            st.sidebar.error(f"Error deleting {file_path}: {e}")
            


# Ensure the script runs properly
if __name__ == "__main__":
    main()