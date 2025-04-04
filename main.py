from fastapi import FastAPI
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
groq_client = Groq()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are CarbonXAgent, an expert on carbon emissions reduction. Try to be concise. "},
                {"role": "user", "content": query}
            ],
            max_tokens=200
        )
        answer = response.choices[0].message.content
        return {"status_code": 200, "response_content": answer}
    except Exception as e:
        return {"status_code": 500, "response_content": f"Error: {str(e)}"}