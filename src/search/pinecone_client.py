import os
import importlib
from typing import Optional


def get_pinecone_index(index_name: Optional[str] = None, create_if_missing: bool = True, dim: Optional[int] = None):
    """Return a Pinecone index-like object while supporting both old and new pinecone package APIs.

    If the package exposes top-level init()/Index(), uses that. Otherwise it will try to
    instantiate pinecone.Pinecone(...) and use pc.Index(name).
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing in environment")

    try:
        pinecone_pkg = importlib.import_module("pinecone")
    except ModuleNotFoundError:
        raise RuntimeError("pinecone package not installed; please add 'pinecone' to requirements.txt")

    # Old-style API: pinecone.init(...); pinecone.Index(name)
    if hasattr(pinecone_pkg, "init"):
        try:
            pinecone_pkg.init(api_key=api_key)
        except Exception:
            # some variants ignore init args
            try:
                pinecone_pkg.init()
            except Exception:
                pass

        if index_name:
            return pinecone_pkg.Index(index_name)
        return pinecone_pkg

    # New-style API: pinecone.Pinecone(...) instance
    if hasattr(pinecone_pkg, "Pinecone"):
        Pinecone = getattr(pinecone_pkg, "Pinecone")
        ServerlessSpec = getattr(pinecone_pkg, "ServerlessSpec", None)
        pc = Pinecone(api_key=api_key)

        if index_name:
            # create index if missing (best-effort)
            try:
                existing = [i.name for i in pc.list_indexes()]
            except Exception:
                existing = []
            if index_name not in existing and create_if_missing:
                if dim is None:
                    dim = int(os.getenv("EMBED_DIM", "3072"))
                try:
                    spec = ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-west-2")) if ServerlessSpec else None
                    pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)
                except Exception:
                    # best-effort create; ignore failures here and let later operations surface them
                    pass
            return pc.Index(index_name)

    raise RuntimeError("Unsupported pinecone package API; upgrade or adjust your pinecone installation")
