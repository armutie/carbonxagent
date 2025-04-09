from fastapi import FastAPI, Form, UploadFile
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from textwrap import dedent
from initiatives.process import process_summary
import json
import chromadb
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document

load_dotenv()

app = FastAPI()
client = chromadb.PersistentClient(path="./chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

core_db = Chroma(client=client, collection_name="core_db", embedding_function=embeddings)
core_retriever = core_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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
    user_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query
    history = request.history
    user_id = request.user_id
    try:
        # Convert history to LangChain message format
        chat_history = []
        for msg in history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # Invoke RAG chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        answer = result["answer"]

        if "FINAL DESCRIPTION:" in answer:
            summary = answer.split("FINAL DESCRIPTION:")[1].strip()
            result = process_summary(summary, user_id)
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
        return {"status_code": 200, "response_content": answer}
    except Exception as e:
        return {"status_code": 500, "response_content": f"Error: {str(e)}"}

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

        # Decode file content and create a LangChain Document
        file_text = file_content.decode("utf-8")
        document = Document(page_content=file_text, metadata={"filename": filename, "user_id": user_id})

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents([document])
        print(f"Split into {len(docs)} chunks")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {len(doc.page_content)} characters")
        db = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
        db.add_documents(docs)

        return {"status_code": 200, "response_content": f"Added {filename} ({len(docs)} chunks) to {'core ' if is_core.lower() == 'true' else ''}datastore"}
    except UnicodeDecodeError:
        return {"status_code": 400, "response_content": "File must be a valid UTF-8 text file"}
    except Exception as e:
        return {"status_code": 500, "response_content": f"Upload error: {str(e)}"}

# RAG endpoint with LangChain retriever
@app.get("/rag")
def get_rag_context(summary: str, user_id: str):
    try:
        user_db = Chroma(client=client, collection_name=f"user_{user_id}", embedding_function=embeddings)
        user_retriever = user_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = user_retriever.get_relevant_documents(summary)
        response_content = "\n".join([doc.page_content for doc in docs]) if docs else ""
        return {"status_code": 200, "response_content": response_content}
    except Exception as e:
        return {"status_code": 500, "response_content": f"Error: {str(e)}"}