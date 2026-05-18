from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import joblib
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY_ENV')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY_ENV')


# print(PINECONE_API_KEY)
# print(GOOGLE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# dumping the embedding
joblib.dump(embeddings, "embeddings.joblib")

bm25_encoder = BM25Encoder().default()
bm25_encoder.fit([t.page_content for t in text_chunks])
bm25_encoder.dump("bm25_values.joblib")
# load the values from the joblib file   



#Initializing the Pinecone
index_name="medical-chatbot"

pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():

    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="dotproduct", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

index = pc.Index(index_name)
retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)                                    
retriever.add_texts([t.page_content for t in text_chunks])

# # query = "What are allergies in human"
# # relevant_docs = retriever.get_relevant_documents(query=query, search_kwargs={'k': 2})

print("Hybrid vectors are available in the vectorstore Pinecone")



