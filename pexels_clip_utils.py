# pexels_clip_utils.py
# -------------------
# Helper: search Pexels for multiple candidates, download them, score with CLIP against segment text,
# and return the best PIL image + scored list.
#
# Usage:
#   from pexels_clip_utils import best_image_from_pexels_for_segment
#   img, scored = best_image_from_pexels_for_segment("A red rocket launches", pexels_api_key="KEY")
#
# Requires: transformers, pillow, requests, torch

import os
import io
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
DEFAULT_PER_PAGE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_proc = None

def _ensure_clip_loaded():
    global _clip_model, _clip_proc
    if _clip_model is None:
        # This will download the CLIP weights the first time; ok in Colab but may take time.
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def pexels_search(api_key, query, per_page=DEFAULT_PER_PAGE, page=1):
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    r = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def download_image(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def extract_query_variants(segment_text):
    """
    Lightweight query expansion. Replace or extend with spaCy/WordNet for better coverage.
    Returns ordered list of query strings.
    """
    text = (segment_text or "").strip()
    if not text:
        return []
    tokens = [w.strip(".,!?()\"'") for w in text.split()]
    stop = set(["the","a","an","in","on","with","is","are","was","were","and","of","to","for","at","that","this","it","its"])
    keywords = [t for t in tokens if t and t.lower() not in stop]
    kw_join = " ".join(keywords[:3]) if keywords else text
    variants = [text]
    if kw_join and kw_join.lower() != text.lower():
        variants.append(kw_join)
    if keywords:
        variants.append(keywords[0])
    # Add style tags that often help Pexels return photographic matches
    variants.append(f"{kw_join} photo")
    variants.append(f"{kw_join} cinematic")
    # Keep order and dedupe
    out = []
    seen = set()
    for v in variants:
        vs = v.lower().strip()
        if vs and vs not in seen:
            seen.add(vs)
            out.append(v)
    return out

def clip_score_text_image(text, pil_image):
    """
    Returns cosine similarity score between text and image using CLIP.
    """
    _ensure_clip_loaded()
    inputs = _clip_proc(text=[text], images=pil_image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = _clip_model(**inputs)
        text_emb = outputs.text_embeds
        img_emb = outputs.image_embeds
        text_emb = text_emb / text_emb.norm(p=2,dim=-1,keepdim=True)
        img_emb = img_emb / img_emb.norm(p=2,dim=-1,keepdim=True)
        sim = (text_emb @ img_emb.T).cpu().numpy()[0,0]
    return float(sim)

def best_image_from_pexels_for_segment(segment_text, pexels_api_key=None, max_candidates=12, per_page=12):
    """
    Returns (best_pil_image, scored_list)
    scored_list: [{'sim': float, 'pil': PIL.Image, 'meta': {url,query,id}} ...] sorted desc by sim.
    Raises ValueError if no API key available.
    """
    if pexels_api_key is None:
        pexels_api_key = os.environ.get("PEXELS_KEY") or os.environ.get("PEXELS_API_KEY")
    if not pexels_api_key:
        raise ValueError("Pexels API key not provided. Set env PEXELS_KEY or pass pexels_api_key.")
    queries = extract_query_variants(segment_text)
    images_meta = []
    for q in queries:
        try:
            resp = pexels_search(pexels_api_key, q, per_page=per_page)
        except Exception as e:
            # network, rate limit or other issues — skip query
            print("Pexels search failed for:", q, str(e))
            continue
        for photo in resp.get("photos", []):
            src = photo.get("src", {})
            url = src.get("large2x") or src.get("large") or src.get("original") or src.get("medium")
            if url:
                images_meta.append({"url": url, "query": q, "id": photo.get("id")})
        if len(images_meta) >= max_candidates:
            break
    images_meta = images_meta[:max_candidates]
    scored = []
    for meta in images_meta:
        try:
            pil = download_image(meta["url"])
            sim = clip_score_text_image(segment_text, pil)
            scored.append({"sim": sim, "pil": pil, "meta": meta})
        except Exception as e:
            print("failed download/score:", meta.get("url"), e)
            continue
    if not scored:
        return None, []
    scored = sorted(scored, key=lambda x: x["sim"], reverse=True)
    return scored[0]["pil"], scored
