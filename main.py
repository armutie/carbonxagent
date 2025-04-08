from fastapi import FastAPI, Form, UploadFile
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from textwrap import dedent
from initiatives.process import process_summary
import json
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer


load_dotenv()
app = FastAPI()
groq_client = Groq()
client = chromadb.PersistentClient(path="./chroma_db")
model = SentenceTransformer('all-mpnet-base-v2')

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]
    user_id: str

conversation_system_prompt = dedent("""
        You’re a sharp assistant gathering detailed info about a company to calculate its carbon emissions. 
Your goal is to collect:
1. What the company does (e.g., logistics, baking).
2. All major emission sources (e.g., diesel trucks, electricity—could be multiple).
3. Specific, quantifiable numbers for each source (e.g., '200 gallons per truck monthly', '1000 kWh monthly').
Chat naturally, asking one focused question at a time based on what’s missing. Start with: “What’s a major emission source in your operations?” after getting the company type. 
Push for numbers (e.g., 'How much diesel?'). If the user’s vague (e.g., ‘lots’), ask ‘Can you estimate a number?’ 
Convert daily to monthly if needed (e.g., 100 gallons/day → 3000 gallons/month). 
The user has the ability to send files to you.
Keep asking ‘Any other sources?’ until they say no. 
ONLY use details the user provides—do NOT invent numbers or sources, even if plausible. Stick strictly to their input unless converting units. 
When you have the company type and at least one quantified source (all mentioned sources need numbers), 
summarize it like: 'A manufacturing plant operating 20 coal-fired furnaces that consume 500 tons of coal per month each, and a fleet of 10 diesel delivery trucks using 400 gallons of diesel per month each.'
Then say 'FINAL DESCRIPTION: [summary]' (no asterisks) to end.
    """)

@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query
    history = request.history
    user_id = request.user_id
    try:
        messages = [{"role": "system", "content": conversation_system_prompt}] + history + [{"role": "user", "content": query}]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=200
        )
        answer = response.choices[0].message.content

        if "FINAL DESCRIPTION:" in answer:
            summary = answer.split("FINAL DESCRIPTION:")[1].strip()
            result = process_summary(summary, user_id)
            try:
                parsed = json.loads(result[0])
                emissions = json.loads(result[1])
                suggestions = json.loads(result[2])
            except Exception as e:
                return {"status_code": 501, "response_content": "Something is wrong with JSON loading"}

            # Format cleanly
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
async def update_vector(user_id: str = Form(...), file: UploadFile = None):
    if not file:
        return {"status_code": 400, "response_content": "No file provided"}
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        return {"status_code": 400, "response_content": "File too large (>10MB)"}
    file_content = await file.read()
    file_text = file_content.decode("utf-8")
    collection = client.get_or_create_collection(f"user_{user_id}")
    embedding = model.encode(file_text).tolist()
    collection.add(
        documents=[file_text],
        embeddings=[embedding],
        ids=[f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
    )

#TODO: Summary might get too long for URL parameters, need to deal with this issue
@app.get("/rag")
def get_rag_context(summary: str, user_id: str):
    collection = client.get_collection(f"user_{user_id}")
    if not collection:
        return {"status_code": 200, "response_content": ""}
    embedding = model.encode(summary).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    response_content = "\n".join(results["documents"][0]) if results["documents"] else ""
    return {"status_code": 200, "response_content": response_content}