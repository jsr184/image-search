#!/usr/bin/env python3
# build-gallery-index.py
# Ingest a local photo folder:
# - Detect objects with torchvision Faster R-CNN (COCO)
# - OCR visible text with EasyOCR
# - Embed with CLIP (open_clip)
# - Store metadata in SQLite and vectors in FAISS
# After it runs you'll have: photos.db, clip.index, faiss_id_map.npy

import os
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch import amp
import torchvision
from torchvision.transforms.functional import to_tensor

import open_clip
import faiss
import easyocr

# absolute!!!
IMAGE_DIR = "/home/jsr184/ai/gallery/testimages"

DB_PATH = "photos.db"
FAISS_PATH = "clip.index"
IDMAP_PATH = "faiss_id_map.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# modify this if you want to make it more aggressive or less
SCORE_THRESH = 0.50  # detector confidence threshold
OCR_MIN_CONF = 0.40  # easyocr text confidence

# Make PIL tolerant of slightly corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# COCO classes with background at idx 0; pad to avoid IndexError
COCO_CLASSES = ["__background__",
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
 "cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
 "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
 "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
 "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
 "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
 "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
 "hair drier","toothbrush"]
while len(COCO_CLASSES) <= 100:
    COCO_CLASSES.append(f"coco_id_{len(COCO_CLASSES)}")

# Supported image file extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------- DB HELPERS -----------------
def connect_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""CREATE TABLE IF NOT EXISTS images(
        id INTEGER PRIMARY KEY, path TEXT UNIQUE, width INT, height INT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS detections(
        image_id INT, label TEXT, score REAL, x1 INT, y1 INT, x2 INT, y2 INT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS ocr(
        image_id INT, text TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        image_id INT, dim INT, vec BLOB)""")
    return con

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# ----------------- MODELS -----------------
@torch.no_grad()
def load_detector():
    # COCO pretrained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def load_clip():
    # You can swap to a LAION checkpoint if you prefer
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(DEVICE)
    return model, preprocess, tokenizer

def load_ocr():
    return easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))


# ----------------- IMAGE / FS -----------------
def iter_images(root: str):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {root}")
    for p in rootp.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p

def safe_open(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ----------------- INFERENCE -----------------
@torch.no_grad()
def detect(det_model, img_pil: Image.Image, score_thresh=SCORE_THRESH):
    img = to_tensor(img_pil).to(DEVICE)
    out = det_model([img])[0]
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy().astype(int)
    scores = out["scores"].detach().cpu().numpy()

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    results = []
    for b, l, s in zip(boxes, labels, scores):
        # Safe label lookup with padding fallback
        name = COCO_CLASSES[l] if 0 <= l < len(COCO_CLASSES) else f"coco_id_{l}"
        results.append((name, float(s), *[int(x) for x in b]))
    return results

@torch.no_grad()
def clip_image_embed(clip_model, preprocess, img_pil: Image.Image) -> np.ndarray:
    img = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        feat = clip_model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ----------------- MAIN BUILD -----------------
def build_index():
    # Normalize working directory to script location for predictable paths
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    con = connect_db()
    det_model = load_detector()
    clip_model, preprocess, _ = load_clip()
    ocr = load_ocr()

    embs: List[np.ndarray] = []
    id_map: List[int] = []

    paths = list(iter_images(IMAGE_DIR))
    if not paths:
        print(f"No images found under: {IMAGE_DIR}")
        con.close()
        return

    for p in tqdm(paths, desc="Processing images"):
        img = safe_open(p)
        if img is None:
            continue

        # Insert or get image row
        cur = con.execute("SELECT id FROM images WHERE path=?", (str(p),))
        row = cur.fetchone()
        if row:
            image_id = row[0]
        else:
            con.execute(
                "INSERT INTO images(path, width, height) VALUES(?,?,?)",
                (str(p), img.width, img.height),
            )
            image_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Skip if we already have an embedding (idempotent)
        if con.execute(
            "SELECT 1 FROM embeddings WHERE image_id=? LIMIT 1", (image_id,)
        ).fetchone():
            continue

        # Object detection (robust to occasional failures)
        try:
            dets = detect(det_model, img)
            if dets:
                con.executemany(
                    "INSERT INTO detections(image_id,label,score,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?)",
                    [(image_id, l, s, x1, y1, x2, y2) for (l, s, x1, y1, x2, y2) in dets],
                )
        except Exception as e:
            # Log and continue — we can still index this image with CLIP
            print(f"[warn] detection failed for {p}: {e}")

        # OCR (best-effort)
        try:
            ocr_res = ocr.readtext(np.array(img))
            if ocr_res:
                text_concat = " ".join([t for _, t, conf in ocr_res if conf >= OCR_MIN_CONF])
                if text_concat.strip():
                    con.execute(
                        "INSERT INTO ocr(image_id, text) VALUES(?,?)",
                        (image_id, text_concat),
                    )
        except Exception as e:
            print(f"[warn] OCR failed for {p}: {e}")

        # CLIP embedding (required)
        vec = clip_image_embed(clip_model, preprocess, img)
        con.execute(
            "INSERT INTO embeddings(image_id, dim, vec) VALUES(?,?,?)",
            (image_id, len(vec), np_to_blob(vec)),
        )

        embs.append(vec)
        id_map.append(image_id)

        # Periodic commits for safety
        if (len(embs) % 256) == 0:
            con.commit()

    con.commit()

    # Build FAISS index from new embeddings (if any)
    if embs:
        D = len(embs[0])
        xb = np.vstack(embs).astype(np.float32)
        # Normalize to ensure cosine via inner product
        norms = np.linalg.norm(xb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        xb = xb / norms

        index = faiss.IndexFlatIP(D)
        index.add(xb)
        faiss.write_index(index, FAISS_PATH)
        np.save(IDMAP_PATH, np.array(id_map, dtype=np.int64))
        print(f"Wrote FAISS index with {len(id_map)} vectors (dim={D}).")
    else:
        print("No new embeddings to index.")

    con.close()


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    print(f"Gallery: {IMAGE_DIR}")
    build_index()
    # Quick post-run hints
    if os.path.exists(FAISS_PATH) and os.path.exists(IDMAP_PATH):
        try:
            import sqlite3
            con = sqlite3.connect(DB_PATH)
            n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            con.close()
            print(f"Embeddings in DB: {n}")
        except Exception:
            pass
