## **Bias Detection Toolkit**  
_A Python toolkit to detect and mitigate bias in text using a BERT-based model._

—

### **🚀 Features**
– Detects potential bias in text using a retrained **BERT model**  
– Provides a **FastAPI**-powered web API  
– Supports both **POST and GET** requests  
– Can be installed as a **Python package** or used as a **CLI tool**  
– Compatible with **PyTorch and Hugging Face Transformers**  

—

### **📂 Project Structure**
```
bias_detection_toolkit/
│── bias_filter/
│   ├── __init__.py
│   ├── bias_filter.py  # Core logic for bias detection
│   ├── model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin  # Trained model weights
│   │   ├── tokenizer.json  # Tokenizer file
│── api/
│   ├── __init__.py
│   ├── main.py  # FastAPI app
│── setup.py  # Packaging configuration
│── requirements.txt
│── README.md
│── tests/
│   ├── test_bias_filter.py
```

—

### **🔧 Installation**

#### **1 Install Locally**
```sh
git clone https://github.com/yourusername/bias-detection.git
cd bias-detection
pip install -e .
```

#### **2 Install via PyPI (if published)**
```sh
pip install bias_detection_toolkit
```

—

### **🚀 Usage**

#### **1 As a Python Module**
```python
from bias_filter.bias_filter import BiasFilter

bias_filter = BiasFilter(model_path="bias_filter/model/")
output = bias_filter.process_statement("[MASK] is the CEO of a company.")
print(output)
```

#### **2 As an API**
Run the FastAPI server:
```sh
uvicorn api.main:app –reload
```
Then, make API calls:

##### **POST request**
```sh
curl -X 'POST' 'http://127.0.0.1:8000/unbias' \
     -H 'Content-Type: application/json' \
     -d '{"text": "[MASK] is the CEO of a multinational corporation."}'
```

##### **GET request**
```sh
curl -X 'GET' 'http://127.0.0.1:8000/unbias?text=[MASK]%20is%20the%20CEO%20of%20a%20company.'
```

—

### **CLI Usage**
After installation, run:
```sh
bias-detect
```

—

### **Development**
#### **Run Tests**
```sh
pytest tests/
```

#### **Build Package**
```sh
python setup.py sdist bdist_wheel
```

#### **Publish to PyPI**
```sh
twine upload dist/*
```

—

### **📜 License**
No License !
