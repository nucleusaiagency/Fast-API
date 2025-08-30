import os
from dotenv import load_dotenv
load_dotenv()

from ..search.pinecone_client import get_pinecone_index


def get_index(create_if_missing: bool = True):
    index_name = os.getenv("PINECONE_INDEX", "transcripts")
    dim = int(os.getenv("EMBED_DIM", "3072"))
    return get_pinecone_index(index_name=index_name, create_if_missing=create_if_missing, dim=dim)
