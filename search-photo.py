import sqlite3, numpy as np, faiss, os
import open_clip, torch
from pathlib import Path

DB_PATH = "photos.db"
FAISS_PATH = "clip.index"
IDMAP_PATH = "faiss_id_map.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(DEVICE)
    return model, tok

@torch.no_grad()
def text_embed(model, tok, text: str):
    t = tok([text]).to(DEVICE)
    with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        e = model.encode_text(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.detach().cpu().numpy().astype(np.float32)

def search(prompt: str, topk=12, must_label=None, ocr_contains=None):
    con = sqlite3.connect(DB_PATH)
    index = faiss.read_index(FAISS_PATH)
    idmap = np.load(IDMAP_PATH)

    model, tok = load_clip()
    q = text_embed(model, tok, prompt)
    D, I = index.search(q, topk*10)   # oversample, then filter

    hits = []
    for score, idx in zip(D[0], I[0]):
        img_id = int(idmap[idx])
        # optional filters
        ok = True
        if must_label:
            ok = con.execute("SELECT 1 FROM detections WHERE image_id=? AND label=? LIMIT 1",
                             (img_id, must_label)).fetchone() is not None
        if ok and ocr_contains:
            row = con.execute("SELECT text FROM ocr WHERE image_id=? LIMIT 1", (img_id,)).fetchone()
            ok = row is not None and (ocr_contains.lower() in row[0].lower())

        if ok:
            path = con.execute("SELECT path FROM images WHERE id=?", (img_id,)).fetchone()[0]
            hits.append({"image_id": img_id, "path": path, "score": float(score)})
        if len(hits) >= topk:
            break

    con.close()
    return hits

if __name__ == "__main__":
    import sys, json
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "a person riding a bike"
    res = search(prompt, topk=8, must_label=None, ocr_contains=None)
    print(json.dumps(res, indent=2))
