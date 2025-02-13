from fastapi import FastAPI
from pydantic import BaseModel
from bias_filter.bias_filter import BiasFilter

# Initialize FastAPI app and BiasFilter model
app = FastAPI()
bias_filter = BiasFilter()

# Define request model
class TextRequest(BaseModel):
    text: str

@app.get("/unbias")
@app.post("/unbias")
async def process_statement(text: str = None, request: TextRequest = None):
    """API endpoint to detect bias in text using both GET and POST"""

    # Handle GET request (text comes from query parameter)
    if text:
        result = bias_filter.process_statement(text)
    
    # Handle POST request (text comes from request body)
    elif request:
        result = bias_filter.process_statement(request.text)
    
    else:
        return {"error": "No input text provided"}

    return {"Final Statement": result}