# src/search/cli.py
import os, argparse, json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import importlib

def build_filter(args):
    f = {}
    if args.program:
        f["program"] = {"$eq": args.program}
    if args.session_name:
        f["session_name"] = {"$eq": args.session_name}
    if args.speakers:
        f["speakers"] = {"$in": args.speakers}
    if args.year_from or args.year_to:
        yr = {}
        if args.year_from: yr["$gte"] = int(args.year_from)
        if args.year_to:   yr["$lte"] = int(args.year_to)
        f["year"] = yr
    # raw key=value filters (e.g. workshop_session_number=1)
    for kv in (args.raw or []):
        k, v = kv.split("=", 1)
        # try int, else string
        try: v = int(v)
        except: v = v
        f[k] = {"$eq": v}
    return f or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query text")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--program", choices=["Podcast","Workshop","MMM","MWM"])
    ap.add_argument("--session-name")
    ap.add_argument("--speakers", nargs="*")
    ap.add_argument("--year-from")
    ap.add_argument("--year-to")
    ap.add_argument("--raw", nargs="*", help="extra raw filters key=value (e.g. workshop_session_number=1)")
    args = ap.parse_args()

    openai_client = OpenAI()
    # Import pinecone at runtime to avoid import-time errors in editors/environments
    try:
        pc_mod = importlib.import_module("pinecone")
    except ModuleNotFoundError:
        raise RuntimeError("pinecone package not installed. Please install the 'pinecone' package in your environment.")
    pc = pc_mod.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX", "transcripts"))

    emb = openai_client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        input=args.q
    ).data[0].embedding

    filt = build_filter(args)
    res = index.query(
        vector=emb,
        top_k=args.k,
        include_metadata=True,
        filter=filt
    )

    for i, m in enumerate(res.get("matches", []), 1):
        md = m.get("metadata", {})
        print(f"\n#{i}  score={m.get('score'):.4f}")
        print(f"   id: {m.get('id')}")
        print(f"   program: {md.get('program')} | year: {md.get('year')} | session: {md.get('session_name')}")
        print(f"   title: {md.get('title')}")
        print("   text:", (md.get("text") or "")[:280].replace("\n"," ") + " ...")

    print("\nFilter used:", json.dumps(filt, indent=2))

if __name__ == "__main__":
    main()
