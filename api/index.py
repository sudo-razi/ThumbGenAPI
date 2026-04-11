from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Query
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io
import os
import httpx
import base64

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Thumbnail Generator API",
    description="Image thumbnail & palette extraction service – Vercel edition",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_KEY", "pixwallapi")   # Set in Vercel → Settings → Environment Variables
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_SIZE_KB = 50           # JPEG binary-search target
MAX_THUMBNAIL_DIM = 800       # Standard thumbnail (longest edge)
PROFILE_THUMBNAIL_DIM = 80   # Profile square crop

# Vercel hard limits
# Request body  : 4.5 MB  (set via vercel.json → functions.bodySizeLimit)
# Response body : 4.5 MB  (platform ceiling, cannot be raised)
VERCEL_RESPONSE_LIMIT_BYTES = 4 * 1024 * 1024   # 4 MB safety margin


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prepare_image(img: Image.Image, target_format: str) -> Image.Image:
    """Prepare image for saving: convert to RGB for JPEG, preserve RGBA for PNG/WebP if needed."""
    if target_format.upper() in ("JPEG", "JPG"):
        if img.mode != "RGB":
            return img.convert("RGB")
    elif target_format.upper() in ("PNG", "WEBP"):
        if img.mode not in ("RGB", "RGBA"):
            return img.convert("RGBA")
    return img


def extract_palette(image_bytes: bytes, num_colors: int = 4) -> list[str]:
    """Return up to `num_colors` dominant hex codes from the image."""
    try:
        img = _prepare_image(Image.open(io.BytesIO(image_bytes)), "JPEG")
        img.thumbnail((100, 100))

        paletted = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=num_colors)
        palette = paletted.getpalette()
        color_counts = paletted.getcolors() or []
        color_counts.sort(key=lambda x: x[0], reverse=True)

        return [
            "#{:02x}{:02x}{:02x}".format(
                palette[idx * 3],
                palette[idx * 3 + 1],
                palette[idx * 3 + 2],
            )
            for _, idx in color_counts[:num_colors]
        ]
    except Exception:
        return []


def process_image(image_bytes: bytes, mode: str = "standard", target_format: str = "JPEG") -> bytes:
    """
    Resize + compress an image.
    target_format: "JPEG", "PNG", or "WEBP"
    """
    try:
        format_upper = target_format.upper()
        if format_upper == "JPG": format_upper = "JPEG"
        
        img = _prepare_image(Image.open(io.BytesIO(image_bytes)), format_upper)

        if mode == "profile":
            img = ImageOps.fit(
                img,
                (PROFILE_THUMBNAIL_DIM, PROFILE_THUMBNAIL_DIM),
                Image.Resampling.LANCZOS,
            )
        else:
            img.thumbnail((MAX_THUMBNAIL_DIM, MAX_THUMBNAIL_DIM), Image.Resampling.LANCZOS)

        final_buf = io.BytesIO()
        
        if format_upper == "PNG":
            # PNG is lossless, no binary search needed
            img.save(final_buf, format="PNG", optimize=True)
        else:
            # Binary-search the quality to hit ≤ TARGET_SIZE_KB (for JPEG and WEBP)
            low, high, best_quality = 1, 95, 75
            
            for _ in range(7):
                mid = (low + high) // 2
                tmp = io.BytesIO()
                img.save(tmp, format=format_upper, quality=mid, optimize=True)
                if tmp.tell() <= TARGET_SIZE_KB * 1024:
                    best_quality = mid
                    low = mid + 1
                    final_buf = tmp
                else:
                    high = mid - 1

            if final_buf.tell() == 0:
                img.save(final_buf, format=format_upper, quality=1, optimize=True)

        result = final_buf.getvalue()

        # Guard against the Vercel 4.5 MB response ceiling
        if len(result) > VERCEL_RESPONSE_LIMIT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Processed image ({len(result) // 1024} KB) exceeds Vercel's "
                    f"4 MB response limit. Upload a smaller source image."
                ),
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")


async def fetch_image_from_url(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            if len(content) > VERCEL_RESPONSE_LIMIT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Remote image is too large (> 4 MB).",
                )
            return content
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to fetch image: {e}",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching URL: {e}")


def _attachment_headers(filename: str) -> dict:
    safe = filename or "image.jpg"
    if "." not in safe:
        safe += ".jpg"
    return {"Content-Disposition": f'attachment; filename="{safe}"'}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/status", tags=["Health"])
async def status():
    return {
        "status": "healthy",
        "service": "thumbnail-generator",
        "target_size": f"{TARGET_SIZE_KB}KB",
        "max_thumbnail_dim": MAX_THUMBNAIL_DIM,
        "profile_thumbnail_dim": PROFILE_THUMBNAIL_DIM,
    }


# -- Upload-based endpoints ----------------------------------------------------

@app.post(
    "/generate_palette",
    tags=["Upload"],
    dependencies=[Depends(get_api_key)],
    summary="Dominant colour palette only",
)
@app.post("/generate_palattle", include_in_schema=False, dependencies=[Depends(get_api_key)])
async def generate_palette(
    file: UploadFile = File(...),
    num_colors: int = Query(4, ge=1, le=4),
):
    _require_image(file)
    content = await file.read()
    palette = extract_palette(content, num_colors=num_colors)
    return {"palette": palette, "count": len(palette)}


@app.post(
    "/generate_thumbnail",
    tags=["Upload"],
    dependencies=[Depends(get_api_key)],
    summary="Standard thumbnail + palette from upload",
)
async def generate_thumbnail(
    file: UploadFile = File(...),
    num_colors: int = Query(4, ge=1, le=4),
    out_format: str = Query("jpeg", description="Output format: jpeg, png, or webp"),
):
    _require_image(file)
    content = await file.read()
    
    fmt = out_format.lower()
    if fmt not in ("jpeg", "jpg", "png", "webp"):
        fmt = "jpeg"
        
    thumbnail_bytes = process_image(content, mode="standard", target_format=fmt)
    palette = extract_palette(content, num_colors=num_colors)
    
    media_types = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = media_types.get(fmt, "image/jpeg")
    
    return {
        "thumbnail": base64.b64encode(thumbnail_bytes).decode("utf-8"),
        "palette": palette,
        "filename": file.filename.rsplit(".", 1)[0] + f".{fmt}",
        "size_bytes": len(thumbnail_bytes),
        "media_type": media_type
    }


@app.post(
    "/generate_profile_thumbnail",
    tags=["Upload"],
    dependencies=[Depends(get_api_key)],
    summary="Square profile thumbnail from upload (binary download)",
)
async def generate_profile_thumbnail(
    file: UploadFile = File(...),
    out_format: str = Query("jpeg", description="Output format: jpeg, png, or webp"),
):
    _require_image(file)
    content = await file.read()
    
    fmt = out_format.lower()
    if fmt not in ("jpeg", "jpg", "png", "webp"):
        fmt = "jpeg"
        
    thumbnail_bytes = process_image(content, mode="profile", target_format=fmt)
    
    media_types = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = media_types.get(fmt, "image/jpeg")
    
    return Response(
        content=thumbnail_bytes,
        media_type=media_type,
        headers=_attachment_headers(file.filename.rsplit(".", 1)[0] + f".{fmt}"),
    )



# -- URL-based endpoints -------------------------------------------------------

@app.post(
    "/generate_thumbnail_url",
    tags=["URL"],
    dependencies=[Depends(get_api_key)],
    summary="Standard thumbnail + palette from a public URL",
)
async def generate_thumbnail_url(
    image_url: str = Query(..., description="Public image URL"),
    num_colors: int = Query(4, ge=1, le=4),
    out_format: str = Query("jpeg", description="Output format: jpeg, png, or webp"),
):
    content = await fetch_image_from_url(image_url)
    
    fmt = out_format.lower()
    if fmt not in ("jpeg", "jpg", "png", "webp"):
        fmt = "jpeg"
        
    thumbnail_bytes = process_image(content, mode="standard", target_format=fmt)
    palette = extract_palette(content, num_colors=num_colors)
    filename = image_url.split("/")[-1].split("?")[0].rsplit(".", 1)[0] + f".{fmt}"
    
    media_types = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = media_types.get(fmt, "image/jpeg")
    
    return {
        "thumbnail": base64.b64encode(thumbnail_bytes).decode("utf-8"),
        "palette": palette,
        "filename": filename,
        "size_bytes": len(thumbnail_bytes),
        "media_type": media_type
    }


@app.post(
    "/generate_profile_thumbnail_url",
    tags=["URL"],
    dependencies=[Depends(get_api_key)],
    summary="Square profile thumbnail from a public URL (binary download)",
)
async def generate_profile_thumbnail_url(
    image_url: str = Query(...),
    out_format: str = Query("jpeg", description="Output format: jpeg, png, or webp"),
):
    content = await fetch_image_from_url(image_url)
    
    fmt = out_format.lower()
    if fmt not in ("jpeg", "jpg", "png", "webp"):
        fmt = "jpeg"
        
    thumbnail_bytes = process_image(content, mode="profile", target_format=fmt)
    filename = image_url.split("/")[-1].split("?")[0].rsplit(".", 1)[0] + f".{fmt}"
    
    media_types = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = media_types.get(fmt, "image/jpeg")

    return Response(
        content=thumbnail_bytes,
        media_type=media_type,
        headers=_attachment_headers(filename),
    )


# ── Guard ──────────────────────────────────────────────────────────────────────

def _require_image(file: UploadFile):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")