import os
from dotenv import load_dotenv

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # Keep these imports
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
# Retrieve the API key explicitly after loading dotenv
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Add a check to ensure the key is loaded
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

def setup_rag_pipeline(pdf_path):
    """
    Sets up the entire RAG pipeline from a given PDF document.
    This includes loading, splitting, embedding, and creating the retriever.
    """
    print("Loading and splitting the document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Document split into {len(splits)} chunks.")

    print("Creating embeddings and vector store...")
    # --- Pass the API key explicitly here ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("Vector store created successfully.")

    retriever = vectorstore.as_retriever()
    
    return retriever

def create_rag_chain(retriever):
    """
    Creates the RAG chain that combines the retriever with a prompt and an LLM.
    """
    print("Creating the RAG chain...")
    # --- Pass the API key explicitly here as well ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    If you don't know the answer, just say that you don't know.
    Don't make up an answer.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain created.")
    
    return retrieval_chain

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    pdf_file_path = "listor.pdf"

    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found. Please place it in the project folder.")
    else:
        rag_retriever = setup_rag_pipeline(pdf_file_path)
        rag_chain = create_rag_chain(rag_retriever)

        print("\n--- You can now ask questions about the document ---")
        
        question = input("Write your question:")
        print(f"\nAsking question: {question}")
        
        response = rag_chain.invoke({"input": question})

        print("\nAnswer:")
        print(response["answer"])

        # question_2 = "Who is the author?"
        # print(f"\nAsking question: {question_2}")
        # response_2 = rag_chain.invoke({"input": question_2})
        # print("\nAnswer:")
        # print(response_2["answer"])