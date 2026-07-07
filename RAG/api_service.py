from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import retrieve_and_generate
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI app
app = FastAPI()

Instrumentator().instrument(app).expose(app)

# Define request and response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

# Load necessary data or models (if any)
# For example, if your RAG pipeline requires preloaded data, load it here
# Example: with open('chunked_data.json') as f:
#              data = json.load(f)

@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    """
    Endpoint to handle queries using the RAG pipeline.
    """
    try:
        # Call the RAG pipeline function from rag.py
        response = retrieve_and_generate(request.query)
        return QueryResponse(query=request.query, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "ok"}
