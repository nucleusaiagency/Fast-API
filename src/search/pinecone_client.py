import os
import importlib
import time
from typing import Optional, Any, Dict


class PineconeConnectionError(Exception):
    """Custom exception for Pinecone connection issues"""
    pass


class PineconeIndexProxy:
    """A small adapter that exposes query/upsert methods and delegates to either the
    old top-level pinecone.Index object or the new Pinecone client instance.
    """
    def __init__(self, *, old_index: Optional[Any] = None, client: Optional[Any] = None, name: Optional[str] = None):
        self._old = old_index
        self._client = client
        self._name = name

    def _handle_operation(self, operation_name: str, operation, **kwargs) -> Dict[str, Any]:
        """Generic handler for retrying operations with error handling"""
        max_retries = 3
        retry_delay = 1  # seconds
        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                return operation(**kwargs)
            except Exception as e:
                last_error = e
                attempts += 1
                if attempts < max_retries:
                    time.sleep(retry_delay * attempts)  # Exponential backoff
                continue

        if "Connection refused" in str(last_error):
            raise PineconeConnectionError(
                f"Failed to {operation_name} after {max_retries} attempts. Service may be starting up."
            )
        raise RuntimeError(f"Operation {operation_name} failed. Error: {str(last_error)}")

    def query(self, **kwargs) -> Dict[str, Any]:
        # Try old-style index API first
        if self._old is not None:
            return self._handle_operation("query", self._old.query, **kwargs)

        # New client-style: try client.query(index=..., ...)
        if self._client is not None:
            try:
                return self._handle_operation(
                    "query",
                    lambda **kw: self._client.query(index=self._name, **kw),
                    **kwargs
                )
            except TypeError:
                return self._handle_operation("query", self._client.query, **kwargs)

        raise RuntimeError("No underlying pinecone index/client available for query")

    def upsert(self, **kwargs) -> Any:
        if self._old is not None:
            return self._old.upsert(**kwargs)
        if self._client is not None:
            try:
                return self._client.upsert(index=self._name, **kwargs)
            except TypeError:
                return self._client.upsert(**kwargs)
        raise RuntimeError("No underlying pinecone index/client available for upsert")

    def delete(self, **kwargs) -> Any:
        if self._old is not None:
            return self._old.delete(**kwargs)
        if self._client is not None:
            try:
                return self._client.delete(index=self._name, **kwargs)
            except TypeError:
                return self._client.delete(**kwargs)
        raise RuntimeError("No underlying pinecone index/client available for delete")

    # passthrough for other behaviors if needed
    def __getattr__(self, item):
        if self._old is not None:
            return getattr(self._old, item)
        if self._client is not None and hasattr(self._client, item):
            return getattr(self._client, item)
        raise AttributeError(item)


def get_pinecone_index(index_name: Optional[str] = None, create_if_missing: bool = True, dim: Optional[int] = None) -> PineconeIndexProxy:
    """Return a PineconeIndexProxy that works with older pinecone modules or the newer Pinecone client.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing in environment")

    try:
        pinecone_pkg = importlib.import_module("pinecone")
    except ModuleNotFoundError:
        raise RuntimeError("pinecone package not installed; please add 'pinecone' to requirements.txt")

    # Old-style API: pinecone.init(...); pinecone.Index(name)
    if hasattr(pinecone_pkg, "init") and hasattr(pinecone_pkg, "Index"):
        try:
            pinecone_pkg.init(api_key=api_key)
        except Exception:
            try:
                pinecone_pkg.init()
            except Exception:
                pass

        if index_name:
            old_index = pinecone_pkg.Index(index_name)
            return PineconeIndexProxy(old_index=old_index, name=index_name)
        return PineconeIndexProxy(old_index=None, name=index_name)

    # New-style API: pinecone.Pinecone(...) instance
    if hasattr(pinecone_pkg, "Pinecone"):
        Pinecone = getattr(pinecone_pkg, "Pinecone")
        ServerlessSpec = getattr(pinecone_pkg, "ServerlessSpec", None)
        pc = Pinecone(api_key=api_key)

        # Ensure index exists (best-effort)
        if index_name:
            try:
                existing = []
                try:
                    existing_raw = pc.list_indexes()
                    # list_indexes may return an object or list
                    if hasattr(existing_raw, "names"):
                        existing = list(existing_raw.names()) if callable(existing_raw.names) else list(existing_raw.names)
                    elif isinstance(existing_raw, (list, tuple)):
                        existing = list(existing_raw)
                except Exception:
                    existing = []

                if index_name not in existing and create_if_missing:
                    if dim is None:
                        dim = int(os.getenv("EMBED_DIM", "3072"))
                    try:
                        spec = ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-west-2")) if ServerlessSpec else None
                        pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)
                    except Exception:
                        pass
            except Exception:
                pass

        return PineconeIndexProxy(client=pc, name=index_name)

    raise RuntimeError("Unsupported pinecone package API; upgrade or adjust your pinecone installation")
