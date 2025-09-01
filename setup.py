from setuptools import setup, find_packages

setup(
    name="transcripts-search",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "openai",
        "pinecone-client",
        "python-dotenv",
    ]
)
