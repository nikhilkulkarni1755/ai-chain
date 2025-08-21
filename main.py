from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="LangChain + OpenAI API", version="1.0.0")

# Initialize OpenAI chat model
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing OpenAI: {e}")
    llm = None

# Request models
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

class ChatResponse(BaseModel):
    response: str
    model: str

# Routes
@app.get("/")
async def root():
    return {"message": "LangChain + OpenAI + FastAPI is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": llm is not None,
        "langchain_version": "0.1.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    try:
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        response = llm.invoke(messages)
        
        return ChatResponse(
            response=response.content,
            model="gpt-3.5-turbo"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/summarize")
async def summarize_text(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    try:
        system_prompt = "You are an expert at summarizing text. Provide a concise summary of the following text."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please summarize this text: {request.message}")
        ]
        
        response = llm.invoke(messages)
        
        return ChatResponse(
            response=response.content,
            model="gpt-3.5-turbo"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
