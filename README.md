# Qwen-Image-Edit on Runpod Serverless

This project implements a [Runpod](https://www.runpod.io/) Serverless worker for running [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), with efficient image caching using Backblaze B2 via MinIO client.

## Features

- **URL-only input**: Process images using publicly accessible URLs (no base64 or file uploads)
- **Smart caching**: Automatically cache source images in Backblaze B2 to avoid repeated downloads
- **Secure storage**: Store edited results in Backblaze B2 with presigned URLs for secure access
- **Memory-only processing**: All image processing happens in memory without disk I/O
- **Comprehensive error handling**: Detailed error responses with taxonomy for debugging
- **Structured logging**: JSON-formatted logs for monitoring and observability
- **Fast model downloads**: Uses `hf-transfer` for accelerated Hugging Face model pulls

## Prerequisites

1. [Runpod](https://www.runpod.io/) account with Serverless enabled
2. [Backblaze B2](https://www.backblaze.com/b2/cloud-storage.html) account with:
   - S3-compatible credentials (Key ID and Application Key)
   - A private bucket created
3. [Hugging Face](https://huggingface.co/) account (optional, for private models)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/qwen-image-edit-runpod.git
   cd qwen-image-edit-runpod
   ```

2. Configure environment variables by copying `.env.example` to `.env` and filling in your values:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

3. Build the Docker image:
   ```bash
   docker build -t qwen-image-edit-runpod .
   ```

4. Push to a container registry that Runpod can access (Docker Hub, GHCR, etc.)

## Deployment to Runpod

1. Create a new Serverless endpoint in Runpod
2. Configure the endpoint with:
   - Container image: Your pushed image
   - Environment variables from your `.env` file
   - GPU type: Select based on model requirements (check model documentation)
3. Deploy the endpoint

## Environment Variables

| Name | Required | Description |
|------|----------|-------------|
| `HF_TOKEN` | Optional | Hugging Face token for private models |
| `S3_ENDPOINT` | Required | Backblaze B2 S3 endpoint (e.g., `s3.us-west-000.backblazeb2.com`) |
| `S3_ACCESS_KEY` | Required | Backblaze Key ID |
| `S3_SECRET_KEY` | Required | Backblaze Application Key |
| `S3_BUCKET` | Required | Backblaze B2 bucket name |
| `S3_REGION` | Optional | Bucket region (default: `us-west-000`) |
| `S3_SECURE` | Optional | Use HTTPS (default: `true`) |
| `S3_OBJECT_PREFIX` | Optional | Prefix for object keys (default: `qwen-image-edit/`) |
| `PRESIGN_EXPIRY` | Optional | Presigned URL expiry in seconds (default: `86400`) |
| `MAX_IMAGE_BYTES` | Optional | Maximum image size in bytes (default: `26214400` = 25MB) |
| `TIMEOUT_SECONDS` | Optional | Operation timeout in seconds (default: `120`) |
| `LOG_LEVEL` | Optional | Logging level (default: `INFO`) |

## Usage

After deploying to Runpod, send a POST request to your endpoint with the following JSON payload:

### Basic Request

```json
{
  "input": {
    "image_url": "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",
    "prompt": "make colors warmer and add gentle film grain"
  }
}
```

### Full Request

```json
{
  "input": {
    "image_url": "https://example.com/img.png",
    "prompt": "remove background and place subject on a studio gray backdrop",
    "negative_prompt": "low quality, artifacts",
    "seed": 424242,
    "num_inference_steps": 40,
    "guidance_scale": 6.5,
    "strength": 0.85,
    "scheduler": "DPMSolverMultistep",
    "output_format": "jpeg",
    "output_quality": 95,
    "safety_filter": true,
    "extra": {
      "keep_resolution": true
    }
  }
}
```

## Response Format

### Success

```json
{
  "status": "success",
  "result": {
    "presigned_url": "https://...",
    "bucket": "YOUR_BUCKET",
    "object_key": "results/<hash>/<jobid>.png",
    "content_type": "image/png",
    "expires_in": 86400,
    "meta": {
      "source_url": "https://example.com/path/to/image.jpg",
      "url_sha256": "<hex>",
      "model": "Qwen/Qwen-Image-Edit",
      "prompt": "add a soft cinematic teal-orange grade, keep composition",
      "negative_prompt": "",
      "seed": 12345,
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "strength": 0.8,
      "scheduler": "EulerAncestral",
      "runtime": {
        "latency_ms_total": 0,
        "latency_ms_download": 0,
        "latency_ms_infer": 0,
        "latency_ms_upload": 0
      },
      "image": {
        "width": 0,
        "height": 0,
        "mode": "RGB",
        "format": "PNG"
      }
    }
  }
}
```

### Error

```json
{
  "status": "error",
  "error": {
    "type": "BadInput | DownloadFailed | CacheReadFailed | ModelError | StorageError",
    "message": "human-readable message",
    "details": {}
  }
}
```

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`

3. Run locally:
   ```bash
   python handler.py
   ```

## Running with Node.js Client

A Node.js script is available to easily submit jobs and monitor their status:

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Create a `prompt.json` file with your request (see examples in the Usage section)
   - If no `seed` is provided, the script will automatically generate a random one

3. Set the required environment variables:
   ```bash
   export RUNPOD_TOKEN="your_runpod_api_token"
   export RUNPOD_ENDPOINT_ID="your_endpoint_id"
   ```

4. Run the script:
   ```bash
   npm run run
   # or
   node scripts/run.js
   ```

The script will submit the job, wait for completion, and display the result URL when finished.

## Project Structure

```
.
├── Dockerfile              # Container definition
├── handler.py              # Main Runpod handler implementation
├── .env                    # Environment variables (not in repo)
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.