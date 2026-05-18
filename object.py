import joblib
from pinecone_text.sparse import BM25Encoder
import sys 

encoder_em = joblib.load("embeddings.joblib")
size_em  = sys.getsizeof(encoder_em )
print("size of embedding:", size_em)

bm25_encoder = BM25Encoder().default()
encoder_bm25 = bm25_encoder.load("bm25_values.joblib")
size_bm25 = sys.getsizeof(encoder_bm25)
print("size of BM25_encodings :", size_bm25)