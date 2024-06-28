import os
import time
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")


# Load the PDF in this section.
def load_document(file):
    from langchain_community.document_loaders import PyPDFLoader
    print(f"Loading {file}")
    data_from_pdf = PyPDFLoader(file).load()
    return data_from_pdf


# data is a list.  Each element is a LangChain Document
# for each page of the PDF with the
# page's content and some metadata
# about where in the document the text came from.
data = load_document("../data/cc.pdf")

# To show how many pages the PDF has.
print(f"Pdf has {len(data)} pages.")


# Split data into chunks.
# Chunk size is by default 256 and Chunk overlap is 64.
def chunk_data(text, chunk_size=256, chunk_overlap=64):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   add_start_index=True)
    all_splits = text_splitter.split_documents(text)
    return all_splits


chunks = chunk_data(data)


def insert_or_fetch_embeddings(index_name, chunks):
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # Create an instance of Vector Store
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name in pc.list_indexes().names():
        print('Deleting existing index...')
        pc.delete_index(index_name)

    print('Creating Index...This may take a while...')
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

    print('Inserting embeddings into Pinecone Vector Database...')
    new_vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

    return new_vector_store


# This will put the vectorized data in the Vector DB
# This step will take some time according to the data size...
index_name = 'chduck'
vector_store = insert_or_fetch_embeddings(index_name, chunks)


def ask_and_get_answer(vector_db, question):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Define in-context examples
    examples = [
        {"question": "What is cloud computing?", "answer": "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the internet ('the cloud') to offer faster innovation, flexible resources, and economies of scale."},
        {"question": "What are the three main types of cloud computing?", "answer": "The three main types of cloud computing are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS)."},
        {"question": "What are some benefits of using cloud computing?", "answer": "Some benefits of using cloud computing include cost savings, scalability, performance, speed, and reliability."},
        {"question": "How does cloud storage work?", "answer": "Cloud storage works by allowing data to be stored on remote servers accessed from the internet. It is maintained, operated, and managed by a cloud storage service provider on storage servers that are built on virtualization techniques."},
        {"question": "What is a hybrid cloud?", "answer": "A hybrid cloud is a computing environment that combines a public cloud and a private cloud by allowing data and applications to be shared between them. By using hybrid cloud, businesses can achieve greater flexibility and more deployment options."},
    ]

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    template = """
    The following are some examples of question and answer pairs about the pdf file:

    {examples_prompt}

    This context is retrieved from related source:
    
    {context}

    If and only if the question clearly states a need to a code in C programming language, 
    then make sure you give the code snippet too. Otherwise chat or give answer as you normally do.
    Question: {q}
    
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=['examples_prompt', 'context', 'q'],
        template=template
    )

    chain = prompt | llm

    answer = chain.invoke({'examples_prompt': "\n\n".join([f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex
                                                            in examples]), 'context': "\n\n".join(retrieved_texts),
                                                            'q': question})
    print(answer.page)
    return answer.content


i = 1
while True:
    print('Write Quit or Exit to stop.')
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Exiting...')
        time.sleep(4)
        break

    ai_answer = ask_and_get_answer(vector_store, q)
    print(f'\nAnswer: {ai_answer}')
    print(f'\n {"-" * 50} \n')
