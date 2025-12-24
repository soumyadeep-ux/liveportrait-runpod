# LivePortrait RunPod Serverless Worker

A RunPod serverless worker for generating talking head videos using LivePortrait.

## Files

- `Dockerfile` - Docker image with ComfyUI + LivePortrait nodes
- `handler.py` - RunPod serverless handler
- `start.sh` - Startup script (ComfyUI + handler)

## Deployment

1. **Push to GitHub:**
   ```bash
   cd scripts/liveportrait-runpod
   git init
   git add .
   git commit -m "LivePortrait RunPod serverless worker"
   git remote add origin https://github.com/soumyadeep-ux/liveportrait-runpod.git
   git push -f origin main
   ```

2. **Create RunPod Serverless Endpoint:**
   - Go to RunPod Console → Serverless → Create Endpoint
   - Select "Custom" template
   - GitHub Repo: `soumyadeep-ux/liveportrait-runpod`
   - Branch: `main`
   - GPU: 48GB recommended (for video processing)
   - Min Workers: 0
   - Max Workers: 3

3. **Wait for build** (10-15 minutes for first build)

## API Usage

### Input Format

```json
{
  "input": {
    "source_image": "base64 or URL",
    "driving_video": "base64 or URL",
    "audio": "optional base64 or URL",
    "flag_relative": true,
    "flag_do_crop": true,
    "flag_pasteback": true,
    "driving_smooth": true,
    "driving_multiplier": 1.0
  }
}
```

### Output Format

```json
{
  "video_base64": "data:video/mp4;base64,..."
}
```

### Example Request

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "source_image_url": "https://example.com/face.jpg",
      "driving_video_url": "https://example.com/driving.mp4"
    }
  }'
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_image` | string | required | Base64 encoded image or URL |
| `driving_video` | string | required | Base64 encoded video or URL |
| `audio` | string | optional | Audio for lip sync |
| `flag_relative` | bool | true | Use relative motion |
| `flag_do_crop` | bool | true | Crop face from image |
| `flag_pasteback` | bool | true | Paste result onto original |
| `driving_smooth` | bool | true | Smooth motion |
| `driving_multiplier` | float | 1.0 | Motion intensity |

## Troubleshooting

### Build Errors

If you see ComfyUI dependency errors, ensure the Dockerfile installs:
- `comfyui-workflow-templates`
- `comfyui-embedded-docs`
- `alembic`

### Timeout Issues

- Increase worker timeout in RunPod settings
- For long videos, use smaller clips (5-10 seconds)

### Memory Issues

- Use 48GB GPU for video processing
- Compress driving videos if needed

## Local Testing

```bash
docker build -t liveportrait-runpod .
docker run -p 7860:7860 --gpus all liveportrait-runpod
```
