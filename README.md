# Qwen-Image-Edit on Runpod Serverless

This project implements a [Runpod](https://www.runpod.io/) Serverless worker for running [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit).

## Features

- **URL-only input**: Process images using publicly accessible URLs (no base64 or file uploads)
- **Secure storage**: Store edited results in Backblaze B2 with presigned URLs for secure access
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

Note: The implementation uses scaled dot product attention by default for better quality output. If this fails to initialize, the application will throw an error rather than falling back to attention slicing.

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

### Full Request with All Parameters

```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "prompt": "Transform the image into an oil painting style with rich textures and vibrant colors",
    "negative_prompt": "low quality, blurry, distorted, ugly, deformed",
    "seed": 123456789,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": "EulerAncestral",
    "extra": {
      "true_cfg_scale": 3.0,
      "height": 1024,
      "width": 1024,
      "num_images_per_prompt": 1
    }
  }
}
```

### Parameter Explanations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | String (URL) | Required | Publicly accessible URL of the image to edit |
| `prompt` | String | Required | Text description of the desired edit |
| `negative_prompt` | String | "" | Text description of what to avoid in the edit |
| `seed` | Integer | Random | Seed for reproducible results (random if not provided or ≤ 0) |
| `num_inference_steps` | Integer | 30 | Number of denoising steps (1-50) |
| `guidance_scale` | Float | 7.5 | How closely to follow the prompt (1.0-20.0) |
| `scheduler` | String | "EulerAncestral" | Scheduler algorithm for denoising |
| `extra` | Object | {} | Additional model parameters |

Note: When a seed is not provided or is set to a value ≤ 0, the system will automatically generate a random seed for the generation process. This seed will be returned in the response metadata, allowing you to reproduce the same result by using that seed in a subsequent request.

### Extra Parameters

The `extra` field accepts additional parameters that can be passed directly to the underlying model. For QwenImageEditPipeline, you can include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `true_cfg_scale` | Float | Enables true classifier-free guidance when > 1.0 (default: 1.0) |
| `height` | Integer | Height of the output image (default: calculated from input) |
| `width` | Integer | Width of the output image (default: calculated from input) |
| `num_images_per_prompt` | Integer | Number of images to generate per prompt (default: 1) |
| `sigmas` | List[Float] | Custom sigmas for the denoising process |
| `generator` | torch.Generator | Random number generator for reproducibility |
| `latents` | torch.Tensor | Pre-generated noisy latents |
| `prompt_embeds` | torch.Tensor | Pre-generated text embeddings |
| `negative_prompt_embeds` | torch.Tensor | Pre-generated negative text embeddings |
| `output_type` | String | Type of output ("pil", "np", or "pt") |
| `return_dict` | Boolean | Whether to return a dictionary or tuple |
| `attention_kwargs` | Dict | Additional attention parameters (e.g., `{"scale": 1.0}` for attention scaling) |
| `callback_on_step_end` | Callable | Callback function at the end of each denoising step |
| `callback_on_step_end_tensor_inputs` | List[String] | Tensor inputs for the callback function |
| `max_sequence_length` | Integer | Maximum sequence length for the prompt (default: 512) |

Example usage with extra parameters:
```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "prompt": "make the image look like a watercolor painting",
    "extra": {
      "true_cfg_scale": 4.0,
      "num_images_per_prompt": 2,
      "height": 768,
      "width": 1024
    }
  }
}
```

### Available Schedulers

The following schedulers are available for use with the `scheduler` parameter:

| Scheduler | Description |
|-----------|-------------|
| `EulerAncestral` | Default scheduler. Good balance of quality and speed |
| `DPMSolverMultistep` | Fast convergence with high quality results |
| `DDIM` | Denoising Diffusion Implicit Models - deterministic sampling |
| `DDPM` | Denoising Diffusion Probabilistic Models - original stochastic sampling |
| `PNDM` | Pseudo Numerical Methods - good for few-step inference |
| `LMSDiscrete` | Linear Multistep Method - stable for various step counts |
| `HeunDiscrete` | Second-order Heun method - higher accuracy |
| `KDPM2Ancestral` | Improved KDPM with ancestral sampling |
| `KDPM2` | Improved KDPM without ancestral sampling |
| `DEISMultistep` | Fast sampling with high-order integration |
| `UniPCMultistep` | Unified Predictor-Corrector framework |

For most use cases, `EulerAncestral` (default) or `DPMSolverMultistep` will give good results. The choice of scheduler can affect both the quality of the output and the processing time.

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
      "true_cfg_scale": 1.0,
      "guidance_scale": 7.5,
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
    "type": "BadInput | DownloadFailed | ModelError | StorageError",
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

## VRAM Optimizations

To reduce GPU memory usage, this implementation includes several optimizations:

### Model Loading Optimizations
- **Float16 Precision**: Uses `torch.float16` instead of `torch.bfloat16` to reduce memory consumption
- **Direct GPU Loading**: Loads the model directly to GPU for optimal performance
- **Scaled Dot Product Attention**: Uses `AttnProcessor2_0` for better quality output (no fallback to attention slicing)
- **VAE Slicing**: Enables VAE slicing for additional memory savings
- **VAE Tiling**: Enables VAE tiling for processing larger images with limited memory

### Inference Optimizations
- **Automatic Memory Cleanup**: Clears GPU cache after each inference operation
- **Inference Step Limiting**: Caps inference steps at 50 to prevent excessive memory usage
- **Scaled Dot Product Attention**: Uses more efficient attention mechanism for better quality output

These optimizations can reduce VRAM usage by 30-50% compared to the standard implementation, allowing the model to run on GPUs with as little as 16GB of VRAM while maintaining good performance.

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