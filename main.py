from fastapi import FastAPI, Form, UploadFile, HTTPException, Depends, Header, status
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from textwrap import dedent
from initiatives.process import process_summary
import json
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader 
import tempfile
import os
from rag import get_retriever, get_chroma_client, get_embeddings
#pip install pypdf, supabase
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase_service: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
# Optional: Anon client if backend needs public API calls
supabase_anon: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

client = get_chroma_client()
embeddings = get_embeddings()
core_retriever = get_retriever("core_db")

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Contextualize question prompt for history-aware retrieval
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
 "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, core_retriever, contextualize_q_prompt)

qa_system_prompt = dedent("""
    You’re a sharp assistant gathering detailed info about a company to calculate its carbon emissions. 
    Your goal is to collect:
    1. What the company does (e.g., logistics, baking).
    2. All major emission sources (e.g., diesel trucks, electricity—could be multiple).
    3. Specific, quantifiable numbers for each source (e.g., '200 gallons per truck monthly', '1000 kWh monthly').
    Chat naturally, asking one focused question at a time based on what’s missing. Start with: “What’s a major emission source in your operations?” after getting the company type. 
    Push for numbers (e.g., 'How much diesel?'). If the user’s vague (e.g., ‘lots’), ask ‘Can you estimate a number?’ 
    Convert daily to monthly if needed (e.g., 100 gallons/day → 3000 gallons/month). 
    Keep asking ‘Any other sources?’ until they say no. 
    ONLY use details the user provides—do NOT invent numbers or sources. Stick strictly to their input unless converting units.

    Here is some context from the core database:
    {context}
    Use this context ONLY if it's relevant to the user's question or to provide more accurate information.

    When you have the company type and at least one quantified source (all mentioned sources need numbers), 
    summarize it like: 'A manufacturing plant operating 20 coal-fired furnaces that consume 500 tons of coal per month each, and a fleet of 10 diesel delivery trucks using 400 gallons of diesel per month each.' 
    Then say 'FINAL DESCRIPTION: [summary]' (no asterisks) to end.
    The user has the ability to send files to you. You do not have access to them while conversing, but once that FINAL_DESCRIPTION trigger hits, you will be able to see user-uploaded files via RAG.
""")
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create chains for RAG
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

async def get_current_user(authorization: str = Header(...)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Use status codes
            detail="Invalid or missing Authorization header (must be 'Bearer token')",
        )
    token = authorization.split(" ")[1] # Extract token after "Bearer "
    try:
        # get_user() with the service key implicitly verifies the token.
        response = supabase_service.auth.get_user(token)
        if not response or not response.user:
             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or user not found")
        return response.user # Return the user object directly
    except Exception as e:
        # Log the actual error for debugging
        print(f"Token validation error: {e}")
        # Provide a generic error to the client
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or authentication error"
        )
    
@app.post("/signup")
async def signup(email: str = Form(...), password: str = Form(...)):
    try:
        response = supabase_anon.auth.sign_up({"email": email, "password": password})
        return {"status_code": 200, "user_id": response.user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        response = supabase_anon.auth.sign_in_with_password({"email": email, "password": password})
        return {
            "status_code": 200,
            "access_token": response.session.access_token,
            "user_id": str(response.user.id)
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)): # user is the User object
    query = request.query
    history = request.history
    user_id = str(user.id)
    try:
        try:
            supabase_service.table("chat_messages").insert({
                "user_id": user_id,
                "role": "user",
                "content": query
            }).execute()
        except Exception as db_error:
            print(f"Error saving user message to DB: {db_error}")
            
        # Convert history to LangChain message format
        chat_history = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role and content: # Basic validation
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))
            else:
                print(f"Skipping invalid history message: {msg}")


        # Invoke RAG chain
        try:
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            answer = result.get("answer", "Sorry, I couldn't generate a response.") # Provide default
        except Exception as rag_error:
             print(f"Error invoking RAG chain: {rag_error}")
             answer = "Sorry, an error occurred while processing your request."

        if "FINAL DESCRIPTION:" in answer:
            summary = answer.split("FINAL DESCRIPTION:")[1].strip()
            result = process_summary(summary, user.id)
            try:
                parsed = json.loads(result[0])
                emissions = json.loads(result[1])
                suggestions = json.loads(result[2])
            except Exception as e:
                return {"status_code": 501, "response_content": "Something is wrong with JSON loading"}

            answer = (
                "Here’s your company’s carbon footprint breakdown:\n\n"
                "### Company Operations\n"
                f"- **Type:** {parsed['company_type']}\n"
                f"- **Emission Sources:** {', '.join([s['type'] for s in parsed['emission_sources']])}\n\n"
                "### Carbon Emissions\n"
                f"- **Total:** {emissions['total_emissions']} {emissions['unit']}\n"
                "- **Breakdown:**\n" + "\n".join([f"  - {b['source']}: {b['emissions']} kg CO2e/month" for b in emissions['breakdown']]) + "\n\n"
                "### Emissions Reduction Initiatives\n" +
                "\n".join([f"- **{s['initiative']}**\n  *{s['description']}*\n  **Impact:** {s['impact']}\n  **Track with:** {', '.join(s['metrics'])}"
                           for s in suggestions])
            )

        try:
            supabase_service.table("chat_messages").insert({
                "user_id": user_id,
                "role": "assistant",
                "content": answer
            }).execute()
        except Exception as db_error:
            print(f"Error saving assistant message to DB: {db_error}")

        return {"status_code": 200, "response_content": answer}
    except HTTPException as he: # Re-raise HTTP exceptions from Depends
        raise he
    except Exception as e:
        print(f"Error in /chat endpoint: {e}") 
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/update_vector")
async def update_vector(user_id: str = Form(...), file: UploadFile = None, is_core: str = Form("false")):
    try:
        if not file:
            return {"status_code": 400, "response_content": "No file provided"}
        if file.size > 10 * 1024 * 1024:
            return {"status_code": 400, "response_content": "File too large (>10MB)"}

        file_content = await file.read()
        filename = file.filename
        collection_name = "core_db" if is_core.lower() == "true" else f"user_{user_id}"
        print(f"Storing in collection: {collection_name}")

        if filename.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()  # Returns a list of Document objects, one per page
                for doc in docs:
                    doc.metadata.update({"filename": filename, "user_id": user_id})
                print(f"Loaded {len(docs)} pages from {filename}")
            finally:
                os.unlink(tmp_path)  
        else:
            # Decode file content and create a LangChain Document
            file_text = file_content.decode("utf-8")
            docs = [Document(page_content=file_text, metadata={"filename": filename, "user_id": user_id})]
            for doc in docs:
                    doc.metadata.update({"filename": filename, "user_id": user_id})
            print(f"Loaded text file {filename} with {len(file_text)} characters")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_docs = text_splitter.split_documents(docs)
        print(f"Split into {len(chunked_docs)} chunks")
        for i, doc in enumerate(chunked_docs):
            print(f"Chunk {i+1}: {len(doc.page_content)} characters")
        db = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
        batch_size = 50 
        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i:i + batch_size]
            print(f"Adding batch {i // batch_size + 1} ({len(batch)} chunks)...")
            db.add_documents(batch)
            print(f"Batch {i // batch_size + 1} added")

        print(f"Finished adding all {len(chunked_docs)} chunks")

        return {"status_code": 200, "response_content": f"Added {filename} ({len(chunked_docs)} chunks) to {'core ' if is_core.lower() == 'true' else ''}datastore"}
    except UnicodeDecodeError:
        print("error 1")
        return {"status_code": 400, "response_content": "File must be a valid UTF-8 text file"}
        
    except Exception as e:
        print(f"error {str(e)}")
        return {"status_code": 500, "response_content": f"Upload error: {str(e)}"}

@app.get("/list_files")
def list_files(collection_name: str):
    try:
        db = Chroma(client=client, collection_name=collection_name)
        results = db.get()
        print(f"Collection {collection_name} has {len(results['documents'])} chunks")
        filenames = set(meta["filename"] for meta in results["metadatas"] if "filename" in meta)
        print(f"Found filenames: {filenames}")
        return {"status_code": 200, "response_content": list(filenames)}
    except Exception as e:
        return {"status_code": 500, "response_content": f"Error: {str(e)}"}
    
@app.post("/authenticate")
async def authenticate(password: str = Form(...)):
    if password == ADMIN_PASSWORD:
        return {"status_code": 200, "admin": True}
    else:
        raise HTTPException(status_code=401, detail="Incorrect password")
    
@app.get("/history")
async def get_history(user: dict = Depends(get_current_user)): # user is now the User object from Supabase
    try:
        messages = supabase_service.table("chat_messages").select("role, content").eq("user_id", str(user.id)).order("created_at").execute()

        # Check for PostgREST errors explicitly if possible 
        if hasattr(messages, 'error') and messages.error:
             print(f"Supabase error fetching history: {messages.error}")
             raise HTTPException(status_code=500, detail="Database error fetching history")

        return {
            "status_code": 200,
            "response_content": messages.data # Directly return the list of dicts
        }
    except HTTPException as he: # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        print(f"Error fetching history: {e}") # Log the error
        # Avoid returning the raw exception string to the client
        return {"status_code": 500, "response_content": "Internal server error fetching history"}
