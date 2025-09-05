# Qwen-Image-Edit on Runpod Serverless - Project Context

## Project Overview

This project implements a Runpod Serverless worker for running Qwen/Qwen-Image-Edit, with efficient image caching using Backblaze B2 via MinIO client. It allows users to edit images using text prompts through a serverless endpoint.

### Key Features
- URL-only input: Process images using publicly accessible URLs
- Smart caching: Automatically cache source images in Backblaze B2 to avoid repeated downloads
- Secure storage: Store edited results in Backblaze B2 with presigned URLs for secure access
- Memory-only processing: All image processing happens in memory without disk I/O
- Comprehensive error handling: Detailed error responses with taxonomy for debugging
- Structured logging: JSON-formatted logs for monitoring and observability
- Fast model downloads: Uses `hf-transfer` for accelerated Hugging Face model pulls
- VRAM optimizations: Implements multiple techniques to reduce GPU memory usage

## Project Structure

```
.
├── Dockerfile              # Container definition
├── handler.py              # Main Runpod handler implementation
├── .env                    # Environment variables (not in repo)
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

## Technologies Used

- **Python**: Main programming language
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face library for diffusion models
- **Qwen-Image-Edit**: Alibaba's image editing model
- **Runpod**: Serverless GPU computing platform
- **MinIO**: S3-compatible client for Backblaze B2 integration
- **Pillow**: Python Imaging Library for image processing
- **Pydantic**: Data validation and settings management

## Important

Refer to @DOCS.md for documentation.

## Core Components

### handler.py
The main application file containing:
- Environment variable configuration
- Input validation with Pydantic models
- Image download and caching logic
- Qwen-Image-Edit model loading and inference
- Result encoding and storage
- Error handling and logging

### Key Functions
- `load_model()`: Loads the Qwen-Image-Edit model with VRAM optimizations
- `download_image()`: Downloads and validates images from URLs
- `run_qwen_edit()`: Executes the image editing process
- `handler()`: Main Runpod entrypoint that orchestrates the workflow

## Environment Variables

Required variables:
- `S3_ENDPOINT`: Backblaze B2 S3 endpoint
- `S3_ACCESS_KEY`: Backblaze Key ID
- `S3_SECRET_KEY`: Backblaze Application Key
- `S3_BUCKET`: Backblaze B2 bucket name

Optional variables:
- `HF_TOKEN`: Hugging Face token for private models
- `S3_REGION`: Bucket region (default: us-west-000)
- `S3_SECURE`: Use HTTPS (default: true)
- `S3_OBJECT_PREFIX`: Prefix for object keys (default: qwen-image-edit/)
- `PRESIGN_EXPIRY`: Presigned URL expiry in seconds (default: 86400)
- `MAX_IMAGE_BYTES`: Maximum image size in bytes (default: 26214400 = 25MB)
- `TIMEOUT_SECONDS`: Operation timeout in seconds (default: 120)
- `LOG_LEVEL`: Logging level (default: INFO)

## API Usage

### Input Format
```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "prompt": "make colors warmer and add gentle film grain",
    "negative_prompt": "low quality, artifacts",
    "seed": 424242,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": "EulerAncestral",
    "output_format": "png",
    "output_quality": 95
  }
}
```

### Response Format
Success:
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
      // Processing metadata
    }
  }
}
```

Error:
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

## Development Workflow

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env` (copy from `.env.example`)

3. Run locally:
   ```bash
   python handler.py
   ```

## VRAM Optimizations

To reduce GPU memory usage, the implementation includes:
- Float16 precision instead of bfloat16
- Direct GPU model loading
- Attention slicing
- Memory efficient attention with xformers
- Automatic memory cleanup after inference
- Inference step limiting (capped at 50 steps)

These optimizations can reduce VRAM usage by 30-50%, allowing the model to run on GPUs with as little as 16GB of VRAM.

## Error Handling

The application implements comprehensive error handling with specific error types:
- BadInput: Invalid input parameters
- DownloadFailed: Issues downloading source images
- CacheReadFailed: Problems accessing cached images
- ModelError: Issues during model inference
- StorageError: Problems with Backblaze B2 storage operations
