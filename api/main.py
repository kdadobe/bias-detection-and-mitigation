from fastapi import FastAPI
from pydantic import BaseModel
from bias_filter.bias_filter import BiasFilter  # Import bias processing logic

# Initialize FastAPI app
app = FastAPI()

# Load the bias filter model
bias_filter = BiasFilter()

# Define request model
class TextRequest(BaseModel):
    text: str

@app.post("/unbias")
@app.get("/unbias")
async def process_statement(text: str = None, request: TextRequest = None):
    """API endpoint to detect bias in text"""
    input_text = text if text else (request.text if request else None)
    if not input_text:
        return {"error": "No text provided"}

    result = bias_filter.process_statement(input_text)
    return {"Final Statement": result}

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Bias Detection API is running!"}
