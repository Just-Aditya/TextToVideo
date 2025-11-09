# app.py
import os
import json
import asyncio
import argparse
from pathlib import Path
import importlib

# Core project utilities (these should exist in your repo)
from utility.script.script_generator import generate_script
from utility.audio.audio_generator import generate_audio
from utility.captions.timed_captions_generator import generate_timed_captions
from utility.render.render_engine import get_output_media  # renderer used later

# NEW: pexels + CLIP helper (you added this file)
from pexels_clip_utils import best_image_from_pexels_for_segment

# ---------- Robust import of background/video search helpers ----------
_bg_mod = None
try:
    _bg_mod = importlib.import_module("utility.video.background_video_generator")
except Exception:
    _bg_mod = None

_get_search_candidates_impl = None
_merge_empty_impl = None

# Try to find likely function names inside background_video_generator
if _bg_mod:
    for name in ("getVideoSearchQueriesTimed", "get_video_search_queries_timed", "getVideoSearchQueries", "get_video_search_queries"):
        _get_search_candidates_impl = getattr(_bg_mod, name, None)
        if _get_search_candidates_impl:
            break
    for name in ("merge_empty_intervals", "merge_empty_intervals_timed", "mergeEmptyIntervals"):
        _merge_empty_impl = getattr(_bg_mod, name, None)
        if _merge_empty_impl:
            break

# If still missing, try the video_search_query_generator module as alternative
if _get_search_candidates_impl is None:
    try:
        alt_mod = importlib.import_module("utility.video.video_search_query_generator")
        for name in ("getVideoSearchQueriesTimed", "get_video_search_queries_timed", "getVideoSearchQueries", "get_video_search_queries"):
            _get_search_candidates_impl = getattr(alt_mod, name, None)
            if _get_search_candidates_impl:
                break
    except Exception:
        _get_search_candidates_impl = None

if _merge_empty_impl is None:
    try:
        alt_mod = importlib.import_module("utility.video.video_search_query_generator")
        for name in ("merge_empty_intervals", "merge_empty_intervals_timed", "mergeEmptyIntervals"):
            _merge_empty_impl = getattr(alt_mod, name, None)
            if _merge_empty_impl:
                break
    except Exception:
        _merge_empty_impl = None

# Final shim wrappers that the rest of app.py will call
def getVideoSearchQueriesTimed(script_text, timed_captions):
    """
    Wrapper that returns search_terms for the video generator.
    If the underlying implementation exists, call it; otherwise, try a best-effort fallback.
    """
    if _get_search_candidates_impl:
        try:
            return _get_search_candidates_impl(script_text, timed_captions)
        except Exception as e:
            print("[WARN] underlying get search impl raised:", e)
            return timed_captions or []
    # Fallback: return timed_captions as-is (best-effort)
    if timed_captions:
        return timed_captions
    return []

def merge_empty_intervals(data):
    if _merge_empty_impl:
        try:
            return _merge_empty_impl(data)
        except Exception as e:
            print("[WARN] underlying merge impl raised:", e)
            return data
    return data

# ----------------- Frame building using Pexels + CLIP -----------------
FRAMES_DIR = Path("frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CLIP_ACCEPT_THRESHOLD = float(os.environ.get("CLIP_ACCEPT_THRESHOLD", 0.18))

def extract_segment_text(segment):
    """
    Normalize segment into text string.
    Accepts string, dict or list/tuple shapes.
    """
    if segment is None:
        return ""
    if isinstance(segment, str):
        return segment
    if isinstance(segment, dict):
        for key in ("text", "query", "caption", "sentence"):
            if key in segment and isinstance(segment[key], str):
                return segment[key]
        # fallback: first string value
        for v in segment.values():
            if isinstance(v, str):
                return v
        return json.dumps(segment)
    if isinstance(segment, (list, tuple)):
        for el in segment:
            if isinstance(el, str):
                return el
        return " ".join(map(str, segment))
    return str(segment)

def build_frame_entries_from_search_terms(search_terms, pexels_key, max_candidates=12):
    """
    For each search-term/segment, pick the best Pexels image (CLIP-ranked).
    Returns a list of frame entries (dicts):
      {'start':..., 'end':..., 'src':..., 'sim':..., 'query_text':...}
    """
    frames = []
    if not search_terms:
        return frames

    for i, seg in enumerate(search_terms):
        seg_text = extract_segment_text(seg)
        print(f"[INFO] Processing segment {i}: {seg_text}")
        try:
            best_img, scored = best_image_from_pexels_for_segment(seg_text, pexels_api_key=pexels_key, max_candidates=max_candidates)
        except Exception as e:
            print(f"[ERROR] Pexels+CLIP search failed for segment '{seg_text}': {e}")
            best_img = None
            scored = []

        out_path = FRAMES_DIR / f"frame_{i:04d}.jpg"

        if best_img is not None and scored:
            top_sim = scored[0]["sim"]
            if top_sim >= CLIP_ACCEPT_THRESHOLD:
                try:
                    best_img.save(out_path)
                    print(f"[✓] Saved frame {out_path} (sim={top_sim:.3f}) for segment: {seg_text}")
                    frames.append({
                        "start": seg.get("start") if isinstance(seg, dict) and "start" in seg else None,
                        "end": seg.get("end") if isinstance(seg, dict) and "end" in seg else None,
                        "src": str(out_path),
                        "sim": float(top_sim),
                        "query_text": seg_text
                    })
                    continue
                except Exception as e:
                    print("[WARN] Saving best image failed:", e)

        # Fallback behavior
        print(f"[INFO] No suitable Pexels image for segment '{seg_text}' (top sim: {scored[0]['sim'] if scored else 'N/A'}). Falling back.")
        fallback_src = None
        if isinstance(seg, dict):
            for k in ("src", "url", "path"):
                if k in seg:
                    fallback_src = seg[k]
                    break
        if fallback_src:
            frames.append({"start": seg.get("start"), "end": seg.get("end"), "src": fallback_src, "sim": scored[0]['sim'] if scored else None, "query_text": seg_text})
            print(f"[INFO] Added fallback src from segment: {fallback_src}")
            continue

        # Last resort: add placeholder
        frames.append({"start": None, "end": None, "src": None, "sim": scored[0]['sim'] if scored else None, "query_text": seg_text})
        print(f"[INFO] Added placeholder frame entry for segment {i}")

    return frames

# --------------------------- Main flow ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a topic.")
    parser.add_argument("topic", type=str, help="The topic for the video")
    args = parser.parse_args()

    SAMPLE_TOPIC = args.topic
    SAMPLE_FILE_NAME = "audio_tts.wav"
    VIDEO_SERVER = "pexel"

    # 1) Generate script
    response = generate_script(SAMPLE_TOPIC)
    print("Script generated:\n", response)

    # 2) Generate audio
    asyncio.run(generate_audio(response, SAMPLE_FILE_NAME))

    # 3) Generate timed captions
    timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
    print("Timed captions:", timed_captions)

    # 4) Build search terms
    search_terms = getVideoSearchQueriesTimed(response, timed_captions)
    print("Search terms:", search_terms)

    # 5) Use Pexels+CLIP pipeline when VIDEO_SERVER is pexel-like
    background_video_urls = None
    if search_terms is not None:
        if VIDEO_SERVER and VIDEO_SERVER.lower().startswith("pexel"):
            pexels_key = os.environ.get("PEXELS_KEY") or os.environ.get("PEXELS_API_KEY")
            if not pexels_key:
                print("[WARN] PEXELS_KEY not set. Falling back to repo generator.")
                try:
                    bg_mod = importlib.import_module("utility.video.background_video_generator")
                    gen_fn = getattr(bg_mod, "generate_video_url", None)
                    if gen_fn:
                        background_video_urls = gen_fn(search_terms, VIDEO_SERVER)
                    else:
                        background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
                except Exception as e:
                    print("[WARN] fallback generate_video_url failed:", e)
                    background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
            else:
                # Build frames dicts via Pexels+CLIP
                frames = build_frame_entries_from_search_terms(search_terms, pexels_key)
                print("Background frames (dicts) built from Pexels (sample):", frames[:3] if isinstance(frames, list) else frames)

                # Convert frames dicts -> ((start, end), src) pairs expected by renderer
                pairs = []
                for idx, f in enumerate(frames):
                    # extract start/end values (may be None)
                    s = f.get("start") if isinstance(f, dict) else None
                    e = f.get("end") if isinstance(f, dict) else None
                    src = f.get("src") if isinstance(f, dict) else f

                    # If start/end are None, try to pull from timed_captions at same index
                    if (s is None or e is None) and isinstance(timed_captions, (list, tuple)) and idx < len(timed_captions):
                        try:
                            interval = timed_captions[idx][0]
                            # interval may be like (start, end)
                            if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                                s = s if s is not None else interval[0]
                                e = e if e is not None else interval[1]
                        except Exception:
                            pass

                    # Normalize src to a file:// absolute URI if it's a local path
                    if src:
                        try:
                            p = Path(src)
                            if not p.is_absolute():
                                p = Path.cwd() / p
                            # convert to file URI (file:///abs/path)
                            src = p.resolve().as_uri()
                        except Exception:
                            # leave src as-is if conversion fails
                            pass

                    pairs.append(((s, e), src))

                background_video_urls = pairs
                print("Background frames converted to pairs (sample):", background_video_urls[:3])
                if not background_video_urls:
                    print("[NOTICE] Pexels pipeline returned no frames — falling back to existing generator.")
                    try:
                        bg_mod = importlib.import_module("utility.video.background_video_generator")
                        gen_fn = getattr(bg_mod, "generate_video_url", None)
                        if gen_fn:
                            background_video_urls = gen_fn(search_terms, VIDEO_SERVER)
                        else:
                            background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
                    except Exception as e:
                        print("[WARN] fallback generate_video_url failed:", e)
                        background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
        else:
            # Non-Pexels path: call existing generator if present
            try:
                bg_mod = importlib.import_module("utility.video.background_video_generator")
                gen_fn = getattr(bg_mod, "generate_video_url", None)
                if gen_fn:
                    background_video_urls = gen_fn(search_terms, VIDEO_SERVER)
                else:
                    background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
            except Exception as e:
                print("[WARN] generate_video_url not available, using search_terms directly:", e)
                background_video_urls = getVideoSearchQueriesTimed(response, timed_captions)
    else:
        print("[INFO] No search_terms generated.")

    # 6) Merge empty intervals if available
    try:
        background_video_urls = merge_empty_intervals(background_video_urls)
    except Exception as e:
        print("[WARN] merge_empty_intervals raised, continuing. Error:", e)

    # 7) Render final media
    if background_video_urls is not None:
        try:
            video = get_output_media(SAMPLE_FILE_NAME, timed_captions, background_video_urls, VIDEO_SERVER)
            print("Rendered video:", video)
        except Exception as e:
            print("[ERROR] get_output_media failed:", e)
            print("[DEBUG] background_video_urls passed to renderer:", background_video_urls)
    else:
        print("[INFO] No video to render.")
