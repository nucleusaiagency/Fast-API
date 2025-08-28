import hashlib

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def sha1_text(text: str) -> str:
    return sha1_bytes(text.encode('utf-8'))
