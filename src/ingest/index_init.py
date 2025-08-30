import os
from dotenv import load_dotenv
load_dotenv()
import importlib

# Pinecone v5 serverless client (import at runtime)
def get_index(create_if_missing: bool = True):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "transcripts")
    dim = int(os.getenv("EMBED_DIM", "3072"))

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing in env")

    try:
        pinecone_mod = importlib.import_module("pinecone")
    except ModuleNotFoundError:
        raise RuntimeError("pinecone package not installed. Please install 'pinecone' in your environment.")

    # Try to get Pinecone client classes
    try:
        Pinecone = getattr(pinecone_mod, "Pinecone")
        ServerlessSpec = getattr(pinecone_mod, "ServerlessSpec")
    except AttributeError:
        # Some Pinecone distributions expose different APIs; attempt direct usage
        Pinecone = getattr(pinecone_mod, "Pinecone", None)
        ServerlessSpec = getattr(pinecone_mod, "ServerlessSpec", None)

    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        if not create_if_missing:
            raise RuntimeError(f"Index {index_name} does not exist")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") if ServerlessSpec else None,
        )
    return pc.Index(index_name)
