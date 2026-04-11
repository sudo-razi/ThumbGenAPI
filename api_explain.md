# Thumbnail Generator API Explanation

Welcome to the **Thumbnail Generator API**. This service is designed for high-performance image processing, specifically tailored for creating thumbnails and extracting color palettes while adhering to Vercel's platform limits.

---

## 1. Authentication

The API is protected by an API Key mechanism.
- **Header Name**: `X-API-KEY`
- **Default Key**: `pixwallapi` (This can be configured via the `API_KEY` environment variable in Vercel).

All processing endpoints (except `/status`) require this header to be present and valid.

---

## 2. Core Features

### 🎨 Palette Extraction
The API can analyze an image and return the top dominant colors in Hex format. It uses an adaptive quantization algorithm to find the most representative colors.

### 🖼️ Multi-Format Support
Every thumbnail endpoint supports the `out_format` parameter:
- **JPEG**: Best for photos, uses lossy compression to hit size targets.
- **PNG**: Lossless, preserves transparency (alpha channel).
- **WebP**: Modern format that supports both transparency and lossy/lossless compression.

### ⚡ Smart Optimization
For **JPEG** and **WebP**, the API performs a **binary search** on the image quality. It tries to get the best possible quality while ensuring the file size stays below **50 KB**. This makes it perfect for fast-loading mobile apps and web galleries.

### 🛡️ Platform Safety
- **Request/Response Limits**: Protects against Vercel's 4.5 MB ceiling by rejecting oversized images early.
- **Memory Efficient**: Uses `Pillow`'s streaming and thumbnailing capabilities to minimize RAM usage.

---

## 3. API Endpoints

### 🩺 Health Check
- **`GET /status`**
  Returns the current configuration (target size, max dimensions, etc.).

### 📤 Upload-Based Endpoints
These endpoints accept a multipart/form-data file upload.

- **`POST /generate_palette`**: Returns only the color palette.
- **`POST /generate_thumbnail`**: Returns a JSON object containing a base64-encoded standard thumbnail and the color palette.
- **`POST /generate_profile_thumbnail`**: Returns a **binary download** of a square-cropped (80x80) profile image.

### 🔗 URL-Based Endpoints
These endpoints fetch an image from a public URL and process it.

- **`POST /generate_thumbnail_url`**: Processes a standard thumbnail and returns JSON with the palette.
- **`POST /generate_profile_thumbnail_url`**: Processes a square profile thumbnail and returns a binary download.

---

## 4. Parameter Reference

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `image_url` | Query String | The public URL of the source image. | *Required* |
| `num_colors` | Query String | Number of dominant colors to extract (1-4). | `4` |
| `out_format` | Query String | Output format: `jpeg`, `png`, or `webp`. | `jpeg` |

---

## 5. Usage Examples

### Generate a WebP Profile Thumbnail from a URL
```bash
curl -X POST "https://your-api.vercel.app/generate_profile_thumbnail_url?image_url=https://example.com/photo.jpg&out_format=webp" \
     -H "X-API-KEY: pixwallapi" \
     --output profile.webp
```

### Get Standard Thumbnail + Palette as JSON
```bash
curl -X POST "https://api-thumbnail.razi.dev/generate_thumbnail_url?image_url=https://example.com/photo.jpg&out_format=png" \
     -H "X-API-KEY: pixwallapi"
```

**Example JSON Response:**
```json
{
  "thumbnail": "iVBORw0KGgoAAAANSUhEUgAA...",
  "palette": ["#2b3e50", "#f39c12", "#e74c3c", "#ecf0f1"],
  "filename": "photo.png",
  "size_bytes": 45210,
  "media_type": "image/png"
}
```
