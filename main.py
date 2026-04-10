from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Query
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response
from PIL import Image, ImageOps
import io
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Thumbnail Generator API")

API_KEY = os.getenv("API_KEY", "pixwallapi")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials"
    )

TARGET_SIZE_KB = 50
MAX_THUMBNAIL_DIM = 800
PROFILE_THUMBNAIL_DIM = 80

def process_image(image_bytes: bytes, mode: str = "standard") -> bytes:
    """
    Process image to match requirements:
    - Output as JPG
    - Targeting ~50KB
    - 'standard' maintains aspect ratio (max 800px)
    - 'profile' center-crops to square (400x400)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (handles RGBA/WebP with alpha)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize logic
        if mode == "profile":
            # Center crop to square and resize
            img = ImageOps.fit(img, (PROFILE_THUMBNAIL_DIM, PROFILE_THUMBNAIL_DIM), Image.Resampling.LANCZOS)
        else:
            # Maintain aspect ratio
            img.thumbnail((MAX_THUMBNAIL_DIM, MAX_THUMBNAIL_DIM), Image.Resampling.LANCZOS)

        # Optimize for file size (target 50KB)
        # We perform a binary search on the JPEG quality parameter
        low, high = 1, 95
        best_quality = 75
        final_output = io.BytesIO()

        # Try to find the best quality that stays under 50KB
        for _ in range(7):
            mid = (low + high) // 2
            temp_out = io.BytesIO()
            img.save(temp_out, format="JPEG", quality=mid, optimize=True)
            size = temp_out.tell()
            
            if size <= TARGET_SIZE_KB * 1024:
                best_quality = mid
                low = mid + 1
                final_output = temp_out
            else:
                high = mid - 1

        # If we didn't find a quality (very unlikely with 50KB target and 800px), 
        # use the lowest possible or force resize smaller (omitted for simplicity unless needed)
        if final_output.tell() == 0:
            img.save(final_output, format="JPEG", quality=1, optimize=True)

        return final_output.getvalue()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.get("/status")
async def status():
    return {
        "status": "healthy",
        "service": "thumbnail-generator",
        "target_size": f"{TARGET_SIZE_KB}KB"
    }

@app.post("/generate_thumbnail", dependencies=[Depends(get_api_key)])
async def generate_thumbnail(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    thumbnail_bytes = process_image(content, mode="standard")
    
    headers = {"Content-Disposition": f"attachment; filename=\"{file.filename}\""}
    return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=headers)

@app.post("/generate_profile_thumbnail", dependencies=[Depends(get_api_key)])
async def generate_profile_thumbnail(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    thumbnail_bytes = process_image(content, mode="profile")
    
    headers = {"Content-Disposition": f"attachment; filename=\"{file.filename}\""}
    return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=headers)

async def fetch_image_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch image from URL: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching image from URL: {str(e)}")

@app.post("/generate_thumbnail_url", dependencies=[Depends(get_api_key)])
async def generate_thumbnail_url(image_url: str = Query(...)):
    content = await fetch_image_from_url(image_url)
    thumbnail_bytes = process_image(content, mode="standard")
    
    filename = image_url.split("/")[-1].split("?")[0] or "thumbnail.jpg"
    if "." not in filename:
        filename += ".jpg"
        
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=headers)

@app.post("/generate_profile_thumbnail_url", dependencies=[Depends(get_api_key)])
async def generate_profile_thumbnail_url(image_url: str = Query(...)):
    content = await fetch_image_from_url(image_url)
    thumbnail_bytes = process_image(content, mode="profile")
    
    filename = image_url.split("/")[-1].split("?")[0] or "profile.jpg"
    if "." not in filename:
        filename += ".jpg"
        
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
