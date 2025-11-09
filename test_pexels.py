# test_pexels.py
from pexels_clip_utils import best_image_from_pexels_for_segment
import os
os.environ.setdefault("PEXELS_KEY", "your_pexels_api_key_here")  # optional if already exported

seg = "A bright red rocket launches into the sky"
img, scored = best_image_from_pexels_for_segment(seg, pexels_api_key=os.environ.get("PEXELS_KEY"))
print("Got candidates:", len(scored))
if scored:
    print("Top similarity:", scored[0]["sim"])
    img.save("test_top.jpg")
    print("Saved test_top.jpg")
else:
    print("No image returned; check PEXELS_KEY and network")
