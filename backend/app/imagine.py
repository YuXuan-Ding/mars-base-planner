"""Grok Imagine: render the planned base on the real terrain.

The page sends a screenshot of the 3D view (real HRSC + MOLA elevation) and what is planned on it:
the bases, their facilities and the road. The prompt is built here from that plan, and Grok's
image-edit endpoint turns the screenshot into a concept render of the base built on that ground.
The key is read from XAI_API_KEY in backend/.env (git-ignored).
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from construct import ai as _ai  # noqa: F401  (importing it loads backend/.env)

XAI_URL = "https://api.x.ai/v1/images/edits"
MODEL = os.environ.get("GROK_IMAGE_MODEL", "grok-imagine-image-2.0")

router = APIRouter(prefix="/imagine", tags=["imagine"])


class Base(BaseModel):
    name: str = Field(max_length=80)
    facilities: list[str] = Field(default_factory=list, max_length=40)


class RenderRequest(BaseModel):
    image: str = Field(description="data:image/jpeg;base64,... screenshot of the 3D view", max_length=12_000_000)
    bases: list[Base] = Field(default_factory=list, max_length=12)
    road: str | None = Field(None, max_length=400, description="what the road is, e.g. '46 km road + 32 km bored'")
    built: bool = False
    style: str | None = Field(None, max_length=200, description="extra direction, e.g. 'at dusk'")


def build_prompt(req: RenderRequest) -> str:
    parts = ["Turn this 3D elevation view of Jezero Crater, Mars into a photorealistic aerial concept render. "
             "Keep the terrain exactly as shown: the same ridges, crater rims, slopes and camera angle. "
             "Rust-colored regolith, scattered rocks, thin dusty atmosphere, butterscotch sky near the horizon."]
    for b in req.bases:
        if b.facilities:
            parts.append(f"At the place labeled '{b.name}', show {b.name} built on flattened, graded ground: "
                         + ", ".join(b.facilities) + ".")
    if req.road:
        parts.append(("A finished" if req.built else "A planned") +
                     f" graded road follows the drawn route line ({req.road}); where it is bored, show tunnel "
                     "portals in the hillside.")
    parts.append("Starship-era technology, plausible engineering, small rovers for scale. "
                 "Replace the map labels and lines with the real structures; no text, no UI overlays.")
    if req.style:
        parts.append(req.style)
    return " ".join(parts)


@router.post("/render")
def render(req: RenderRequest):
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise HTTPException(503, "Grok is not set up: add XAI_API_KEY to backend/.env and restart the backend")
    if not req.image.startswith("data:image/"):
        raise HTTPException(422, "image must be a data:image/... URI")
    prompt = build_prompt(req)
    try:
        r = httpx.post(XAI_URL, headers={"Authorization": f"Bearer {key}"}, timeout=180, json={
            "model": MODEL, "prompt": prompt, "image": {"url": req.image, "type": "image_url"}})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach Grok: {type(e).__name__}") from e
    if r.status_code != 200:
        try:
            msg = r.json().get("error") or r.text
        except ValueError:
            msg = r.text
        raise HTTPException(502, f"Grok returned {r.status_code}: {str(msg)[:300]}")
    item = (r.json().get("data") or [{}])[0]
    url = item.get("url") or (f"data:image/jpeg;base64,{item['b64_json']}" if item.get("b64_json") else None)
    if not url:
        raise HTTPException(502, "Grok returned no image")
    return {"url": url, "prompt": prompt, "model": MODEL}
