### About the Project
The overall intention of this repo is to show a `RAG` implementation together with `in-context learning`.

We are using pinecone vector database and `GPT-3.5-Turbo`.

### Prerequisites
Ensure that you have installed the libraries in `requirements.txt` which is located in the `.\Team10\source\requirements.txt`.
You can run this code from terminal:
```py
!pip install -r requirements.txt
```
### Steps to Run
The code is ready for use, you can run it, and you will be prompted to ask questions about the dummy data.
The dummy data is a PDF about cloud computing located in the `.\Team10\data\cc.pdf`.

> [!WARNING]
> Please use your own OpenAI API Key in Line 32 of the code, since the one in the code does not work due to security policies. 
