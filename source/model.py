import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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


def ask_and_get_answer(vector_db):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    system_prompt = ("""
    The following are some examples of question and answer pairs about the pdf file:

    Q1: What is cloud computing?, Answer: Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the internet ('the cloud') to offer faster innovation, flexible resources, and economies of scale.
    Q2: What are the three main types of cloud computing? Answer: The three main types of cloud computing are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).
    Q3: What are some benefits of using cloud computing? Answer: Some benefits of using cloud computing include cost savings, scalability, performance, speed, and reliability.
    Q4: How does cloud storage work? Answer: Cloud storage works by allowing data to be stored on remote servers accessed from the internet. It is maintained, operated, and managed by a cloud storage service provider on storage servers that are built on virtualization techniques.
    Q5: What is a hybrid cloud? Answer: A hybrid cloud is a computing environment that combines a public cloud and a private cloud by allowing data and applications to be shared between them. By using hybrid cloud, businesses can achieve greater flexibility and more deployment options.

    This context is retrieved from related source:

    {context}

    If and only if the question clearly states a need to a code in C programming language, 
    then make sure you give the code snippet too. Otherwise chat or give answer as you normally do.
    """)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Statefully manage chat history
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    i = 1
    while True:
        print('Write Quit or Exit to stop.')
        q = input(f'Question #{i}: ')
        i = i + 1
        if q.lower() in ['quit', 'exit']:
            print('Exiting...')
            time.sleep(4)
            break

        print(f'\nAnswer:')
        print(conversational_rag_chain.invoke(
            {"input": q},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"])
        print(f'\n {"-" * 50} \n')


ask_and_get_answer(vector_store)
