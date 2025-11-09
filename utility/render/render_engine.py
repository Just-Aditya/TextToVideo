# utility/render/render_engine.py
import time
import os
import tempfile
import zipfile
import platform
import subprocess
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip,
                            TextClip, VideoFileClip)
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.audio_normalize import audio_normalize
import requests
from urllib.parse import urlparse, unquote


def download_http_to_file(url, filename):
    """Download an HTTP(S) resource to filename (streamed)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return filename


def resolve_source_to_local_path(src):
    """
    Resolve a source (http(s) URL, file:// URI or local path) into an actual local filesystem
    path that moviepy/ffmpeg can read.

    Returns (local_path, was_downloaded)
      - local_path: absolute local path to the file
      - was_downloaded: True if we downloaded an HTTP resource and should delete it later
    """
    if not src:
        raise ValueError("Empty source passed to resolve_source_to_local_path")

    parsed = urlparse(src)

    # HTTP or HTTPS: download to a temporary file and return its path
    if parsed.scheme in ("http", "https"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(parsed.path)[1] or "")
        tmp.close()
        download_http_to_file(src, tmp.name)
        return os.path.abspath(tmp.name), True

    # file:// URI: convert to local path and return
    if parsed.scheme == "file":
        # For file:///abs/path -> parsed.path contains the absolute path
        path = unquote(parsed.path)
        if parsed.netloc:
            # handle possible netloc (windows drive letters or hosts)
            path = os.path.join(parsed.netloc, path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"file URI points to missing file: {path}")
        return path, False

    # No scheme: treat as local path (absolute or relative)
    # If it exists on disk, return its absolute path
    if os.path.exists(src):
        return os.path.abspath(src), False

    # Try to expand user and relative paths
    alt = os.path.abspath(os.path.expanduser(src))
    if os.path.exists(alt):
        return alt, False

    # Nothing found
    raise FileNotFoundError(f"Source not found or unsupported scheme: {src}")


def download_file(url, filename):
    """
    Backwards-compatible download_file that previously existed.
    Now implemented using resolve_source_to_local_path when possible.
    If url is an HTTP URL, stream-download into filename.
    If url is local or file://, copy or read into filename.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return download_http_to_file(url, filename)
    if parsed.scheme == "file":
        # copy file from file:// URI to filename
        src_path = unquote(parsed.path)
        if parsed.netloc:
            src_path = os.path.join(parsed.netloc, src_path)
        src_path = os.path.abspath(src_path)
        with open(src_path, "rb") as r, open(filename, "wb") as w:
            w.write(r.read())
        return filename
    # treat as local path: copy to filename
    if os.path.exists(url):
        with open(url, "rb") as r, open(filename, "wb") as w:
            w.write(r.read())
        return filename
    raise FileNotFoundError(f"Cannot download or copy source: {url}")


def search_program(program_name):
    try:
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None


def get_program_path(program_name):
    program_path = search_program(program_name)
    return program_path


def get_output_media(audio_file_path, timed_captions, background_video_data, video_server):
    OUTPUT_FILE_NAME = "rendered_video.mp4"
    magick_path = get_program_path("magick")
    # set IMAGEMAGICK_BINARY for moviepy/TextClip
    if magick_path:
        os.environ['IMAGEMAGICK_BINARY'] = magick_path
    else:
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'

    visual_clips = []
    # track temp files we created so we can cleanup them (only files we downloaded)
    temp_files_to_cleanup = []

    # background_video_data expected as iterable of ((start,end), src)
    for (t1, t2), video_src in background_video_data:
        try:
            local_path, was_downloaded = resolve_source_to_local_path(video_src)
        except Exception as e:
            print(f"[WARN] Could not resolve background source '{video_src}': {e}. Skipping this entry.")
            continue

        # If resolved path is not a media file (e.g., None), skip
        if not local_path:
            continue

        # keep track of downloaded temps
        if was_downloaded:
            temp_files_to_cleanup.append(local_path)

        # Create VideoFileClip from the local file path
        try:
            video_clip = VideoFileClip(local_path)
        except Exception as e:
            print(f"[WARN] MoviePy failed to open '{local_path}': {e}. Skipping.")
            continue

        # If t1/t2 are None, don't set start/end (let moviepy place them)
        if t1 is not None:
            try:
                video_clip = video_clip.set_start(float(t1))
            except Exception:
                pass
        if t2 is not None:
            try:
                video_clip = video_clip.set_end(float(t2))
            except Exception:
                pass

        visual_clips.append(video_clip)

    # Add captions as TextClip overlays
    audio_clips = []
    audio_file_clip = AudioFileClip(audio_file_path)
    audio_clips.append(audio_file_clip)

    for (t1, t2), text in timed_captions:
        try:
            text_clip = TextClip(txt=text, fontsize=100, color="white", stroke_width=3, stroke_color="black", method="label")
            if t1 is not None:
                text_clip = text_clip.set_start(t1)
            if t2 is not None:
                text_clip = text_clip.set_end(t2)
            text_clip = text_clip.set_position(["center", 800])
            visual_clips.append(text_clip)
        except Exception as e:
            print(f"[WARN] Failed to create TextClip for '{text}': {e}")

    # if no visual clips, create a blank background from the first frame of audio (fallback)
    if not visual_clips:
        print("[WARN] No visual clips were created; adding a blank ImageClip fallback.")
        h, w = 720, 1280
        blank = ImageClip(color=(0, 0, 0), size=(w, h)).set_duration(audio_file_clip.duration)
        visual_clips.append(blank)

    video = CompositeVideoClip(visual_clips)

    if audio_clips:
        audio = CompositeAudioClip(audio_clips)
        video.duration = audio.duration
        video.audio = audio

    # write final video
    video.write_videofile(OUTPUT_FILE_NAME, codec='libx264', audio_codec='aac', fps=25, preset='veryfast')

    # Clean up downloaded temporary files
    for p in temp_files_to_cleanup:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return OUTPUT_FILE_NAME
