import faiss
import hnswlib

def index_faiss(database):
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    return index

def index_hnsw(database):
    p = hnswlib.Index(space="cosine", dim=database.shape[-1])
    p.init_index(max_elements=20000, ef_construction=200, M=48)
    p.add_items(database)
    return p