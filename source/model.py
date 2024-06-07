import os
from dotenv import load_dotenv, find_dotenv
load_dotenv('.env')
os.environ.get('PINECONE_API_KEY')
os.environ.get('OPENAI_API_KEY')
# Load the PDF in this section.

def load_document(file):
    from langchain.document_loaders import PyPDFLoader
    print(f"Loading {file}")
    data = PyPDFLoader(file).load()
    return data
data = load_document("../data/cc.pdf")

# To show how many pages the PDF has.
print(f"Pdf has {len(data)} pages.")

# Split data into chunks.
# Chunksize is by default 256 and Chunkoverlap is 64.
def chunk_data(data, chunk_size=256):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = 64)
  chunks = text_splitter.split_documents(data)
  return(chunks)
chunks = chunk_data(data)

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key='sk-proj-JnjTpBTJJ1aTBQfMRtJjT3BlbkFJ2N4I9vSeTD6WN6niNlRF') # Best performance according to cost
    # pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    pc = Pinecone(
        api_key='656f5292-ed1e-40ff-a6ad-0b53adb69e51'
    )
    from pinecone import Pinecone
    index_name = "chatdoc" # or your index name
    if index_name in pc.list_indexes().names():
        pc.delete_index("chatdoc")

    # You should use the same similarity metric used to train the model that created the embeddings.
    # For example, if you're using OpenAI (any of their GPT models so far) you should use cosine similarity.
    # https://www.pinecone.io/learn/vector-similarity/

    if 'my_index' not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

# This will put the vectorized data in the Vector DB
# This step will take some time according to the data size...
index_name = 'chatdoc'
vector_store = insert_or_fetch_embeddings(index_name, chunks)

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain import PromptTemplate
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Define in-context examples
    examples = [
        {"question": "What is cloud computing?", "answer": "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the internet ('the cloud') to offer faster innovation, flexible resources, and economies of scale."},
        {"question": "What are the three main types of cloud computing?", "answer": "The three main types of cloud computing are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS)."},
        {"question": "What are some benefits of using cloud computing?", "answer": "Some benefits of using cloud computing include cost savings, scalability, performance, speed, and reliability."},
        {"question": "How does cloud storage work?", "answer": "Cloud storage works by allowing data to be stored on remote servers accessed from the internet. It is maintained, operated, and managed by a cloud storage service provider on storage servers that are built on virtualization techniques."},
        {"question": "What is a hybrid cloud?", "answer": "A hybrid cloud is a computing environment that combines a public cloud and a private cloud by allowing data and applications to be shared between them. By using hybrid cloud, businesses can achieve greater flexibility and more deployment options."},
    ]

    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(q)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    template = """
    The following are some examples of question and answer pairs about cloud computing:

    {examples_prompt}

    Context:
    {context}

    Now, answer the following question:
    Question: {q}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables = ['examples_prompt', 'context', 'q'],
        template = template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run({'examples_prompt':"\n\n".join([f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples]), 'context':"\n\n".join(retrieved_texts), 'q':q})
    return answer

import time
i = 1
print('Write Quit or Exit to quit.')
while True:
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Quitting...')
        time.sleep(2)
        break
    answer = ask_and_get_answer(vector_store, q)
    print(f'\nAnswer: {answer}')
    print(f'\n {"-" * 50} \n')
