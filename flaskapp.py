import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify,CORS
from threading import Lock
from functools import wraps
import jwt

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_POSTGRES_CONNECTION_STRING = os.getenv("SUPABASE_POSTGRES_CONNECTION_STRING")
SUPABASE_PG_COLLECTION_NAME = os.getenv("SUPABASE_PG_COLLECTION_NAME", "document_chunks")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it.")
if not SUPABASE_POSTGRES_CONNECTION_STRING:
    raise ValueError("SUPABASE_POSTGRES_CONNECTION_STRING environment variable not set. Please set it.")

app = Flask(__name__)

# Global variables for RAG components
rag_retriever = None
rag_chain = None
# Global variable for the PGVector instance
vectorstore_instance = None 

initialization_lock = Lock()

def verify_supabase_jwt(token):
    try:
        # For production, it's safer to fetch the JWKS from Supabase (SUPABASE_URL/auth/v1/keys)
        # and verify the token against the public keys.
        # Using the direct JWT secret is simpler for development/smaller apps, but be careful.
        decoded_token = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated", # This is the default audience for Supabase user JWTs
            options={"verify_exp": True} # Verify expiration
        )
        return decoded_token
    except jwt.ExpiredSignatureError:
        print("JWT has expired!")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid JWT: {e}")
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Authorization token is missing!"}), 401

        try:
            token = auth_header.split(" ")[1]
            user_info = verify_supabase_jwt(token)
            if user_info is None:
                return jsonify({"message": "Token is invalid or expired!"}), 401

            request.user_info = user_info # User info is now accessible in your endpoint function
        except IndexError:
            return jsonify({"message": "Token format is invalid (e.g., 'Bearer token')."}), 401
        except Exception as e:
            return jsonify({"message": f"Token error: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated

def get_embeddings_model():
    """Initializes and returns the Google Generative AI Embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def get_llm_model():
    """Initializes and returns the Google Generative AI Chat model."""
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

def initialize_rag_pipeline_for_querying():
    """
    Initializes the RAG pipeline components for querying.
    This assumes the PGVector index already exists and connects to it.
    This runs once on Flask app startup.
    """
    global rag_retriever, rag_chain, vectorstore_instance

    if rag_retriever and rag_chain:
        print("RAG query pipeline already initialized.")
        return

    with initialization_lock:
        if rag_retriever and rag_chain: # Double check in case another thread already initialized
            print("RAG query pipeline already initialized by another thread.")
            return

        print("Initializing RAG query pipeline (connecting to PGVector)...")
        embeddings = get_embeddings_model()

        # Connect to an existing PGVector store for querying
        vectorstore_instance = PGVector(
            collection_name=SUPABASE_PG_COLLECTION_NAME,
            embedding_function=embeddings,
            connection_string=SUPABASE_POSTGRES_CONNECTION_STRING
        )
        print("Connected to PGVector successfully for querying.")

        # --- Create the Retriever ---
        rag_retriever = vectorstore_instance.as_retriever()
        
        # --- Create the RAG Chain ---
        print("Creating the RAG chain...")
        llm = get_llm_model()

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
        rag_chain = create_retrieval_chain(rag_retriever, document_chain)
        print("RAG query pipeline initialized.")

def index_and_add_document_to_pgvector(pdf_path, overwrite_collection=False):
    """
    Processes a PDF, generates embeddings, and adds them to PGVector.
    Can either append to an existing collection or overwrite it.
    """
    print(f"Loading and splitting document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Document split into {len(splits)} chunks.")

    embeddings = get_embeddings_model()

    if overwrite_collection:
        # This will DELETE the existing collection before adding new data. Use with caution!
        print(f"OVERWRITING existing Supabase PGVector collection: {SUPABASE_PG_COLLECTION_NAME}...")
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=SUPABASE_PG_COLLECTION_NAME,
            connection_string=SUPABASE_POSTGRES_CONNECTION_STRING,
            pre_delete_collection=True # Confirms overwrite behavior
        )
    else:
        # This will ADD documents to the existing collection.
        print(f"Adding documents to existing Supabase PGVector collection: {SUPABASE_PG_COLLECTION_NAME}...")
        
        # First, connect to the existing collection
        existing_vectorstore = PGVector(
            collection_name=SUPABASE_PG_COLLECTION_NAME,
            embedding_function=embeddings, # Important: Must provide embedding_function for add_documents
            connection_string=SUPABASE_POSTGRES_CONNECTION_STRING
        )
        
        # Then, add the new documents
        existing_vectorstore.add_documents(splits)
        vectorstore = existing_vectorstore # Ensure vectorstore_instance points to the updated one

    print("Documents embedded and stored in PGVector successfully.")
    # After adding documents, we need to ensure the global retriever is updated
    # This might require re-initializing rag_retriever if PGVector doesn't auto-update it
    global rag_retriever
    if vectorstore:
        rag_retriever = vectorstore.as_retriever()
        print("RAG retriever updated with new documents.")


# --- Flask Routes ---

@app.route('/')
def home():
    """Basic endpoint to check if the API is running."""
    return "RAG Flask API is running!"

@app.route('/ask', methods=['POST'])
@token_required
def ask_question():
    """
    Endpoint to ask questions about the document.
    Expects a JSON body: {"question": "Your question here"}
    """
    # Ensure pipeline is initialized
    if rag_chain is None:
        initialize_rag_pipeline_for_querying() # In case it didn't initialize for some reason
        if rag_chain is None: # Still none? Something is wrong.
            return jsonify({"error": "RAG pipeline failed to initialize."}), 503

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Please provide a 'question' in the request body."}), 400

    print(f"\nReceived question: {question}")
    try:
        response = rag_chain.invoke({"input": question})
        answer = response["answer"]
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        return jsonify({"error": "An error occurred while processing your question."}), 500

@app.route('/index_document', methods=['POST'])
@token_required
def index_document_endpoint():
    """
    Endpoint to trigger document indexing.
    Expects a JSON body:
    {"pdf_path": "/path/to/your/document.pdf", "overwrite": true/false}
    If 'overwrite' is true, existing documents will be deleted before new ones are added.
    """
    data = request.get_json()
    pdf_path = data.get("pdf_path")
    overwrite = data.get("overwrite", False) # Default to adding, not overwriting

    if not pdf_path:
        return jsonify({"error": "Please provide a 'pdf_path' in the request body."}), 400

    if not os.path.exists(pdf_path):
        return jsonify({"error": f"PDF file not found at '{pdf_path}'. Ensure it's accessible on the server."}), 404

    try:
        index_and_add_document_to_pgvector(pdf_path=pdf_path, overwrite_collection=overwrite)
        return jsonify({"message": "Document indexing/addition started/completed successfully."}), 200
    except Exception as e:
        print(f"Error during document indexing: {e}")
        return jsonify({"error": "An error occurred during document indexing."}), 500

# --- Flask App Initialization ---
# This decorator ensures the RAG pipeline is initialized once when the app starts.
# This connects to the existing PGVector index for querying.
@app.before_first_request
def startup_load_pipeline_for_querying():
    """
    Initializes the RAG pipeline on Flask app startup.
    This will connect to the existing PGVector index for querying.
    """
    print("Running @before_first_request to initialize RAG pipeline for querying...")
    initialize_rag_pipeline_for_querying()

if __name__ == '__main__':
    # For local development:
    # 1. Ensure your .env is set up.
    # 2. Run this file: python app.py
    # 3. Open another terminal.
    # 4. FIRST, populate your Supabase PGVector:
    #    curl -X POST -H "Content-Type: application/json" -d '{"pdf_path": "listor.pdf", "overwrite": true}' http://127.0.0.1:5000/index_document
    #    (Replace "listor.pdf" with the actual path if it's not in the same directory as app.py)
    #    This will create/overwrite your collection.
    # 5. THEN, to add another document:
    #    curl -X POST -H "Content-Type: application/json" -d '{"pdf_path": "another_document.pdf", "overwrite": false}' http://127.0.0.1:5000/index_document
    # 6. Finally, you can ask questions:
    #    curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}' http://127.0.0.1:5000/ask

    app.run(debug=True, host='0.0.0.0', port=5000)