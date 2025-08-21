# main.py - Updated with chains
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from typing import Dict

# Load environment variables
load_dotenv()

app = FastAPI(title="LangChain + OpenAI API with Chains", version="1.0.0")

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

class ChainRequest(BaseModel):
    topic: str

class ConversationRequest(BaseModel):
    message: str
    conversation_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    model: str

# Store conversation chains by ID
conversation_chains: Dict[str, LLMChain] = {}

# ===== CHAIN DEFINITIONS =====

def analyze_topic_chain():
    """Modern chain using LCEL"""
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the key concepts of {topic} in exactly 3 bullet points."
    )
    
    critique_prompt = ChatPromptTemplate.from_template(
        "Given this summary: {summary}\n\n"
        "What are 2 potential criticisms or limitations of this topic? "
        "Be constructive and specific."
    )
    
    synthesis_prompt = ChatPromptTemplate.from_template(
        "Summary: {summary}\n"
        "Criticisms: {criticisms}\n\n"
        "Now provide a balanced conclusion that acknowledges both "
        "the strengths and limitations. End with one actionable insight."
    )
    
    # Build the chain using LCEL
    chain = (
        {"topic": RunnablePassthrough()}
        | {"topic": lambda x: x["topic"], 
           "summary": summary_prompt | llm | StrOutputParser()}
        | {"topic": lambda x: x["topic"],
           "summary": lambda x: x["summary"],
           "criticisms": critique_prompt | llm | StrOutputParser()}
        | synthesis_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def parallel_analysis_chain():
    """Run multiple analyses in parallel, then combine results"""
    pros_prompt = ChatPromptTemplate.from_template(
        "List 3 main advantages or benefits of {topic}. Be specific."
    )
    
    cons_prompt = ChatPromptTemplate.from_template(
        "List 3 main disadvantages or challenges with {topic}. Be realistic."
    )
    
    examples_prompt = ChatPromptTemplate.from_template(
        "Provide 2 real-world examples where {topic} is successfully applied."
    )
    
    # Run analyses in parallel
    parallel_chain = RunnableParallel(
        pros=pros_prompt | llm | StrOutputParser(),
        cons=cons_prompt | llm | StrOutputParser(),
        examples=examples_prompt | llm | StrOutputParser()
    )
    
    # Combine results
    combination_prompt = ChatPromptTemplate.from_template(
        "Based on this analysis:\n\n"
        "PROS: {pros}\n\n"
        "CONS: {cons}\n\n"
        "EXAMPLES: {examples}\n\n"
        "Write a balanced assessment that weighs these factors and "
        "provides a recommendation for when this topic/approach is most valuable."
    )
    
    # Full chain: parallel analysis → combination
    full_chain = parallel_chain | combination_prompt | llm | StrOutputParser()
    
    return full_chain

def create_conversation_chain():
    """Chain with memory for multi-turn conversations"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tutor. Use the conversation history to provide contextual responses."),
        ("human", "{chat_history}"),
        ("human", "{input}")
    ])
    
    chain = LLMChain(
        llm=llm,
        prompt=conversation_prompt,
        memory=memory,
        verbose=True
    )
    
    return chain

# ===== ORIGINAL ROUTES =====
@app.get("/")
async def root():
    return {"message": "LangChain + OpenAI + FastAPI with Chains is running!"}

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
        from langchain.schema import HumanMessage, SystemMessage
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

# ===== NEW CHAIN ROUTES =====

@app.post("/chain/analysis")
async def modern_analysis_endpoint(request: ChainRequest):
    """Modern LCEL chain for topic analysis"""
    if llm is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    try:
        chain = analyze_topic_chain()
        result = chain.invoke({"topic": request.topic})
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chain/parallel")
async def parallel_analysis_endpoint(request: ChainRequest):
    """Parallel processing chain"""
    if llm is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    try:
        chain = parallel_analysis_chain()
        result = chain.invoke({"topic": request.topic})
        return {"assessment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chain/conversation")
async def conversation_endpoint(request: ConversationRequest):
    """Conversation chain with memory"""
    if llm is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    try:
        # Get or create conversation chain for this ID
        if request.conversation_id not in conversation_chains:
            conversation_chains[request.conversation_id] = create_conversation_chain()
        
        chain = conversation_chains[request.conversation_id]
        result = chain.invoke({"input": request.message})
        
        return {"response": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)