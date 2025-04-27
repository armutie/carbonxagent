from fastapi import FastAPI, Form, UploadFile, HTTPException, Depends, Header, status, Query
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
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader 
import tempfile
import os
from rag import get_retriever, get_vector_store, get_embeddings, RAGConfig
#pip install pypdf, supabase
from supabase import create_client, Client, AuthApiError

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase_service: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
# Optional: Anon client if backend needs public API calls
supabase_anon: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

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
        user = response.user
        if not response or not response.user:
             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or user not found")
        
        try:
            # Query the user_roles table using the validated user's ID
            status_response = supabase_service.table("user_roles") \
                .select("is_approved") \
                .eq("user_id", str(user.id)) \
                .single() \
                .execute()

            # Check if we got data and if the user is approved
            is_approved = False # Default to not approved
            if status_response.data:
                is_approved = status_response.data.get("is_approved", False)

            # If not approved, raise Forbidden error
            if not is_approved:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account requires admin approval for access. (thru get_current_user)"
                )

        except HTTPException as he:
             raise he # Re-raise the 403 if not approved
        except Exception as db_error:
             # Handle errors fetching the role/status (e.g., DB connection issue)
            print(f"Error checking approval status for user {user.id}: {db_error}")
            # Deny access if status cannot be confirmed
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not verify user approval status."
            )

        return user # Return the user object directly
    except Exception as e:
        # Log the actual error for debugging
        print(f"Token validation error: {e}")
        # Provide a generic error to the client
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or authentication error"
        )
    
async def is_admin(user_id: str) -> bool:
    """Checks if the given user_id has the 'admin' role."""
    try:
        # Use the service client which bypasses RLS for this check
        response = supabase_service.table("user_roles") \
            .select("role") \
            .eq("user_id", user_id) \
            .single() \
            .execute() # Use single() because user_id is unique

        # Debugging: print(f"Role check for {user_id}: {response}")

        # Check if data exists and role is 'admin'
        if response.data and response.data.get("role") == "admin":
            return True
        return False
    except Exception as e:
        # Log error fetching role
        print(f"Error checking admin role for user {user_id}: {e}")
        # Default to False if error occurs or user/role not found
        return False

    
@app.post("/signup")
async def signup(email: str = Form(...), password: str = Form(...)):
    try:
        response = supabase_anon.auth.sign_up({"email": email, "password": password})
        new_user_id = response.user.id

        try:
            # Use the SERVICE client to insert the role
            insert_response = supabase_service.table("user_roles").insert({
                "user_id": new_user_id,
                "role": "user",  # Assign the default role
                "is_approved": False
            }).execute()

            # Optional: Check for errors during role insertion
            if hasattr(insert_response, 'error') and insert_response.error:
                print(f"Error inserting default role for {new_user_id}: {insert_response.error}")
                # Decide how to handle this: Log it? Raise an error?
                # For now, we might let signup succeed but log the role issue.

        except Exception as role_insert_error:
            print(f"Failed to insert default role for user {new_user_id}: {role_insert_error}")
            # Log the error, but potentially allow signup to appear successful

        return {"status_code": 200, "user_id": response.user.id, "message": "Signup successful. Please check your email for confirmation."}

    except AuthApiError as e:
        # Check if the error message indicates a duplicate user
        # Common messages include "User already registered", "duplicate key value violates unique constraint"
        # Inspect the actual error message 'e.message' or 'str(e)' during testing if unsure
        if "User already registered" in e.message or "already exists" in e.message:
             raise HTTPException(
                 status_code=409, # Conflict status code
                 detail="Email already registered. Please try logging in."
             )
        else:
            # Handle other authentication errors
            raise HTTPException(status_code=400, detail=f"{e.message}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        response = supabase_anon.auth.sign_in_with_password({"email": email, "password": password})
        # Bit of a hack, approval doesnt usually happen in login, and sort of makes get_current_user code redundant 
        try:
            status_response = supabase_service.table("user_roles") \
                .select("is_approved") \
                .eq("user_id", str(response.user.id)) \
                .single() \
                .execute()

            is_approved = False
            if status_response.data:
                is_approved = status_response.data.get("is_approved", False)

            if not is_approved:
                # Raise 403 Forbidden HERE if not approved
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account requires admin approval for access. (thru login)"
                )

        except HTTPException as he:
            raise he # Re-raise 403
        except Exception as db_error:
            print(f"Error checking approval status during login for {response.user.id}: {db_error}")
            raise HTTPException(status_code=500, detail="Could not verify user approval status during login.")

        return {
            "status_code": 200,
            "access_token": response.session.access_token,
            "user_id": str(response.user.id),
            "user_email": response.user.email
        }
    except AuthApiError as e:
         # Common message for invalid login: "Invalid login credentials"
        raise HTTPException(status_code=401, detail=f"Login failed: {e.message}")
    except HTTPException as he:
        raise he
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
async def update_vector(file: UploadFile = None, is_core: str = Form("false"), user: dict = Depends(get_current_user)):
    user_id = str(user.id)
    # --- Admin Check ---
    if is_core.lower() == "true":
            admin_status = await is_admin(user_id)
            if not admin_status:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Need to be admin")

    # --- File Validation ---
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    if file.size > 10 * 1024 * 1024: # Example: 10MB limit
        raise HTTPException(status_code=413, detail="File too large (>10MB)")

    try:
        file_content = await file.read()
        filename = file.filename
        # --- Determine NAMESPACE for Pinecone ---
        namespace = "core_db" if is_core.lower() == "true" else f"user_{user_id}"
        print(f"Authenticated user {user_id} preparing to store in Pinecone namespace: {namespace}")

        docs = []
        # --- Load File Content ---
        if filename.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                docs.extend(loaded_docs) # Use extend to add all pages
                print(f"Loaded {len(loaded_docs)} pages from PDF: {filename}")
            finally:
                os.unlink(tmp_path)
        elif filename.lower().endswith(".txt"):
             try:
                file_text = file_content.decode("utf-8")
                doc = Document(page_content=file_text) 
                docs.append(doc)
                print(f"Loaded text file {filename} with {len(file_text)} characters")
             except UnicodeDecodeError:
                print("Unicode decode error for TXT file")
                raise HTTPException(status_code=400, detail="File must be a valid UTF-8 text file")
        else:
             raise HTTPException(status_code=415, detail="Unsupported file type. Please upload PDF or TXT.")

        # --- Add Metadata (CRITICAL FOR PINEONE) ---
        for doc in docs:
             doc.metadata = {
                 "filename": filename,
                 "user_id": user_id,
                 "original_collection": namespace # Store the intended collection as metadata too
                 # Add other metadata as needed (e.g., page for PDF: 'page': doc.metadata.get('page', 0) )
             }
             # Add page number if it exists from PDF loader
             if 'page' in getattr(doc, 'metadata', {}):
                 doc.metadata['page'] = doc.metadata.get('page')


        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract document content.")

        # --- Split Documents ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunked_docs = text_splitter.split_documents(docs)
        print(f"Split into {len(chunked_docs)} chunks")

        try:
            pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
            if not pinecone_index_name:
                raise ValueError("PINECONE_INDEX_NAME env var is missing!")
            print(f"[UpdateVector] Target Pinecone Index: '{pinecone_index_name}'")
            print(f"[UpdateVector] Target Namespace: '{namespace}'")

            # *** Get store for the specific namespace ***
            print(f"[UpdateVector] Getting vector store for namespace '{namespace}'...")
            vector_store = get_vector_store(namespace=namespace)
            print(f"[UpdateVector] Vector store object obtained: {type(vector_store)}")

            # *** Define Batch Size ***
            # Adjust this based on typical chunk size and metadata size.
            # Start smaller (e.g., 50-100) and increase if uploads are too slow but below 2MB.
            batch_size = 100
            print(f"[UpdateVector] Using batch size: {batch_size}")

            total_batches = (len(chunked_docs) + batch_size - 1) // batch_size
            print(f"[UpdateVector] Adding {len(chunked_docs)} chunks in {total_batches} batches...")

            for i in range(0, len(chunked_docs), batch_size):
                batch_num = (i // batch_size) + 1
                batch_docs = chunked_docs[i:i + batch_size]
                print(f"[UpdateVector] Processing Batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)...")

                # *** Add the current BATCH ***
                try:
                    vector_store.add_documents(batch_docs)
                    print(f"[UpdateVector] Batch {batch_num} added successfully.")
                except Exception as batch_error:
                    print(f"---! ERROR adding Batch {batch_num} !---")
                    print(f"Error Type: {type(batch_error).__name__}")
                    print(f"Error Details: {batch_error}")
                    raise Exception(f"Failed on batch {batch_num}") from batch_error

                # Optional: Add a small delay between batches if needed (e.g., for rate limits, though less common for upsert size)
                # time.sleep(0.5)

            print(f"[UpdateVector] All batches processed.")

        except Exception as pinecone_process_error:
            # Catch errors from getting store or during batch loop
            print(f"---! ERROR during Pinecone processing !---")
            print(f"Error Type: {type(pinecone_process_error).__name__}")
            print(f"Error Details: {pinecone_process_error}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed during Pinecone processing: {pinecone_process_error}")

        try:
            print(f"[UpdateVector] Adding file record to Supabase table 'uploaded_files'...")
            # Ensure user_id and namespace are the ones determined earlier
            # Ensure filename is available
            # Ensure chunked_docs list is available for len()

            insert_data = {
                "user_id": user_id,       # User who uploaded
                "filename": filename,     # Name of the file
                "namespace": namespace,   # 'core_db' or 'user_{user_id}'
                # Optional fields:
                # "file_size": file.size, # Get size before reading content if possible
                "chunk_count": len(chunked_docs)
            }
            response = supabase_service.table("uploaded_files").insert(insert_data).execute()

            # More robust error checking for Supabase insert
            if len(response.data) == 0: # Check if data was actually returned (indicates success)
                # Attempt to extract error if available (structure might vary)
                error_message = "Unknown error during Supabase insert"
                if hasattr(response, 'error') and response.error:
                    error_message = str(response.error)
                elif hasattr(response, 'message') and response.message:
                    error_message = str(response.message)

                print(f"!!! WARNING: Failed to insert file record into Supabase: {error_message} !!!")
                # Decide how to handle this - maybe don't fail the whole upload? Log it.
            else:
                print(f"[UpdateVector] Successfully added record to 'uploaded_files': {response.data}")

        except Exception as supabase_insert_error:
            print(f"!!! WARNING: Exception inserting file record into Supabase: {supabase_insert_error} !!!")
            # Log this error but likely continue, as the main upload to Pinecone succeeded.
            import traceback
            print(traceback.format_exc())
        # --- END: Add record to Supabase ---


        print(f"Finished adding all {len(chunked_docs)} chunks for {filename} to namespace '{namespace}'")
        return {"status_code": 200, "response_content": f"Added {filename} ({len(chunked_docs)} chunks) to namespace '{namespace}'"}

    # --- Error Handling ---
    except HTTPException as he:
         raise he 
    except UnicodeDecodeError:
        print(f"Error decoding file {filename}")
        raise HTTPException(status_code=400, detail="File must be a valid UTF-8 text file")
    except Exception as e:
        # Catch potential Pinecone client errors too
        print(f"Unexpected error processing file {filename}: {str(e)}")
        import traceback
        print(traceback.format_exc()) # Good for debugging Pinecone errors
        raise HTTPException(status_code=500, detail=f"File upload processing error: {str(e)}")

    
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
    
@app.get("/my_role")
async def get_my_role(user: dict = Depends(get_current_user)):
    """Fetches the role for the currently authenticated user."""
    user_id = str(user.id)
    is_user_admin = await is_admin(user_id) # Reuse the helper function
    role = "admin" if is_user_admin else "user" # Determine role (can be more complex if >2 roles)
    return {"status_code": 200, "role": role}

@app.get("/list_files")
async def list_files(
    # Use 'namespace' as the query parameter name for clarity,
    # but we'll map it from 'collection_name' in the frontend call later.
    namespace: str = Query(..., alias="collection_name"), # Expect 'collection_name' from frontend
    user: dict = Depends(get_current_user) # Returns Supabase user object
):
    user_id = str(user.id) # Get user ID string

    # Security check: Only admins can list 'core_db' namespace.
    is_user_admin = await is_admin(user_id)
    if namespace == "core_db" and not is_user_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required to list core files")

    # RLS policies handle access control automatically for SELECT based on user role and namespace.
    # Admins can query 'core_db'. Any user can query their own user_{id} namespace (implicitly allowed by RLS).
    # Users attempting to query another user's namespace will get an empty result due to RLS.

    print(f"User {user_id} attempting to list files for namespace: {namespace}")

    try:
        response = supabase_service.table("uploaded_files")\
            .select("filename")\
            .eq("namespace", namespace)\
            .order("filename")\
            .execute()

        # Check for explicit PostgREST errors
        if hasattr(response, 'error') and response.error:
            print(f"Supabase error fetching file list: {response.error}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error fetching file list")

        if response.data:
            filenames = sorted(list({item['filename'] for item in response.data if item.get('filename')}))
            print(f"Found filenames for namespace '{namespace}': {filenames}")
            return {"status_code": 200, "response_content": filenames}
        else:
            print(f"No files found in Supabase table for namespace: {namespace}")
            return {"status_code": 200, "response_content": []} # Return empty list

    except HTTPException as he:
        raise he 
    except Exception as e:
        print(f"Error listing files from Supabase 'uploaded_files' table for namespace '{namespace}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server error listing files")

