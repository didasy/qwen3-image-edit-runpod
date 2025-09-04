import os
import io
import json
import time
import uuid
import hashlib
import logging
from typing import Tuple, Optional
from urllib.parse import urlparse
from datetime import timedelta

import torch
import requests
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel, HttpUrl, validator
from minio import Minio
from minio.error import S3Error
import runpod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Environment variables
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_REGION = os.getenv("S3_REGION", "us-west-000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_SECURE = os.getenv("S3_SECURE", "true").lower() == "true"
S3_OBJECT_PREFIX = os.getenv("S3_OBJECT_PREFIX", "")
PRESIGN_EXPIRY = int(os.getenv("PRESIGN_EXPIRY", "86400"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "26214400"))  # 25 MB
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate required environment variables
required_env_vars = [
    ("S3_ENDPOINT", S3_ENDPOINT),
    ("S3_ACCESS_KEY", S3_ACCESS_KEY),
    ("S3_SECRET_KEY", S3_SECRET_KEY),
    ("S3_BUCKET", S3_BUCKET),
]

for name, value in required_env_vars:
    if not value:
        raise ValueError(f"Environment variable {name} is required")

# Initialize MinIO client
minio_client = Minio(
    S3_ENDPOINT,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    region=S3_REGION,
    secure=S3_SECURE,
)

# Check if bucket exists
try:
    if not minio_client.bucket_exists(S3_BUCKET):
        raise ValueError(f"Bucket {S3_BUCKET} does not exist")
except S3Error as e:
    raise ValueError(f"Failed to access bucket {S3_BUCKET}: {e}")

# Input validation schema
class ImageEditInput(BaseModel):
    image_url: HttpUrl
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.8
    scheduler: str = "EulerAncestral"
    output_format: str = "png"
    output_quality: int = 95
    safety_filter: bool = True
    extra: dict = {}

    @validator("image_url")
    def validate_image_url(cls, v):
        parsed = urlparse(str(v))
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("URL must be HTTP or HTTPS")
        return v

    @validator("output_format")
    def validate_output_format(cls, v):
        if v not in ["png", "jpeg"]:
            raise ValueError("output_format must be 'png' or 'jpeg'")
        return v

    @validator("output_quality")
    def validate_output_quality(cls, v):
        if v < 1 or v > 100:
            raise ValueError("output_quality must be between 1 and 100")
        return v

# Global model variable
model = None

def load_model():
    """Load the Qwen-Image-Edit model once at cold start"""
    global model
    if model is None:
        logger.info("Loading Qwen-Image-Edit model...")
        try:
            # Import required modules
            from diffusers import StableDiffusionInstructPix2PixPipeline, DPMSolverMultistepScheduler
            import torch
            
            # Try to load the Qwen model first
            try:
                model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    token=HF_TOKEN,
                )
                logger.info("Successfully loaded Qwen/Qwen-Image-Edit model")
            except Exception as e:
                logger.warning(f"Failed to load Qwen/Qwen-Image-Edit model: {e}")
                logger.info("Falling back to timbrooks/instruct-pix2pix model")
                # Fallback to the original InstructPix2Pix model
                model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    token=HF_TOKEN,
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            # Use DPMSolverMultistepScheduler for better results
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
            
            model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to CPU if CUDA is not available
            try:
                from diffusers import StableDiffusionInstructPix2PixPipeline
                import torch
                
                # Try to load the Qwen model first
                try:
                    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        "Qwen/Qwen-Image-Edit",
                        torch_dtype=torch.float32,
                        token=HF_TOKEN,
                    )
                    logger.info("Successfully loaded Qwen/Qwen-Image-Edit model on CPU")
                except Exception as e:
                    logger.warning(f"Failed to load Qwen/Qwen-Image-Edit model on CPU: {e}")
                    logger.info("Falling back to timbrooks/instruct-pix2pix model on CPU")
                    # Fallback to the original InstructPix2Pix model
                    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        "timbrooks/instruct-pix2pix",
                        torch_dtype=torch.float32,
                        token=HF_TOKEN,
                    )
                
                model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
                model.eval()
                logger.info("Model loaded on CPU successfully")
            except Exception as e2:
                logger.error(f"Failed to load model on CPU: {e2}")
                raise
    return model

def sha256_hex(text: str) -> str:
    """Calculate SHA256 hash of a string and return hex representation"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_image_extension(content_type: str, url: str) -> str:
    """Determine image extension from content-type or URL"""
    content_type_map = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
    }
    
    if content_type in content_type_map:
        return content_type_map[content_type]
    
    # Fallback to URL extension
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        return "jpg"
    elif path.endswith(".png"):
        return "png"
    elif path.endswith(".webp"):
        return "webp"
    elif path.endswith(".bmp"):
        return "bmp"
    elif path.endswith(".tiff") or path.endswith(".tif"):
        return "tiff"
    
    # Default fallback
    return "png"

def validate_content_type(content_type: str) -> bool:
    """Validate that content type is an allowed image type"""
    allowed_types = [
        "image/jpeg",
        "image/jpg", 
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff"
    ]
    return content_type.lower() in allowed_types

def download_image(url: str) -> Tuple[bytes, str, str]:
    """Download image from URL with validation"""
    logger.info(f"Downloading image from {url}")
    
    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Invalid URL scheme. Only HTTP and HTTPS are supported.")
    
    try:
        # Stream download with timeout and size limits
        response = requests.get(
            url,
            stream=True,
            timeout=(5, 30),  # 5s connect, 30s read
            headers={"User-Agent": "Qwen-Image-Edit-Runpod/1.0"}
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        if not validate_content_type(content_type):
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image too large: {content_length} bytes")
        
        # Download image data
        image_bytes = b""
        for chunk in response.iter_content(chunk_size=8192):
            image_bytes += chunk
            if len(image_bytes) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image exceeded maximum size of {MAX_IMAGE_BYTES} bytes")
        
        # Determine extension
        extension = get_image_extension(content_type, url)
        
        logger.info(f"Downloaded image: {len(image_bytes)} bytes, type: {content_type}, ext: {extension}")
        return image_bytes, extension, content_type
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")

def encode_image(image: PILImage, format: str, quality: int = 95) -> Tuple[bytes, str, str]:
    """Encode PIL image to bytes with specified format"""
    buffer = io.BytesIO()
    
    if format == "jpeg":
        # Convert RGBA to RGB if needed for JPEG
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = background
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        content_type = "image/jpeg"
        extension = "jpg"
    else:  # PNG
        image.save(buffer, format="PNG", optimize=True)
        content_type = "image/png"
        extension = "png"
    
    image_bytes = buffer.getvalue()
    buffer.close()
    
    logger.info(f"Encoded image: {len(image_bytes)} bytes, format: {format}")
    return image_bytes, extension, content_type

def run_qwen_edit(model, image: PILImage, prompt: str, **kwargs) -> PILImage:
    """Run Qwen-Image-Edit on the input image"""
    logger.info(f"Running Qwen-Image-Edit with prompt: {prompt}")
    logger.info(f"Model parameters: {kwargs}")
    
    try:
        # Extract parameters
        negative_prompt = kwargs.get("negative_prompt", "")
        seed = kwargs.get("seed", None)
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        image_guidance_scale = kwargs.get("image_guidance_scale", 1.5)
        strength = kwargs.get("strength", 0.8)
        scheduler = kwargs.get("scheduler", "EulerAncestral")
        safety_filter = kwargs.get("safety_filter", True)
        
        # Set scheduler if specified
        if scheduler == "DPMSolverMultistep":
            from diffusers import DPMSolverMultistepScheduler
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
        elif scheduler == "DDIM":
            from diffusers import DDIMScheduler
            model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
        elif scheduler == "DDPM":
            from diffusers import DDPMScheduler
            model.scheduler = DDPMScheduler.from_config(model.scheduler.config)
        elif scheduler == "PNDM":
            from diffusers import PNDMScheduler
            model.scheduler = PNDMScheduler.from_config(model.scheduler.config)
        elif scheduler == "LMSDiscrete":
            from diffusers import LMSDiscreteScheduler
            model.scheduler = LMSDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler == "HeunDiscrete":
            from diffusers import HeunDiscreteScheduler
            model.scheduler = HeunDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler == "KDPM2Ancestral":
            from diffusers import KDPM2AncestralDiscreteScheduler
            model.scheduler = KDPM2AncestralDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler == "KDPM2":
            from diffusers import KDPM2DiscreteScheduler
            model.scheduler = KDPM2DiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler == "DEISMultistep":
            from diffusers import DEISMultistepScheduler
            model.scheduler = DEISMultistepScheduler.from_config(model.scheduler.config)
        elif scheduler == "UniPCMultistep":
            from diffusers import UniPCMultistepScheduler
            model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
        # EulerAncestral is the default scheduler
        
        # Set generator for reproducible results if seed is provided
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=model.device).manual_seed(seed)
        
        # Run the model
        result = model(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        )
        
        # Return the edited image
        if hasattr(result, 'images') and result.images:
            return result.images[0]
        elif isinstance(result, list) and len(result) > 0:
            return result[0]
        else:
            # Fallback to original image if model didn't return an image
            logger.warning("Model didn't return an edited image, returning original")
            return image
            
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        # Return original image if there's an error
        return image

def handler(event):
    """Main handler function for Runpod serverless"""
    start_time = time.time()
    job_id = event.get("id", str(uuid.uuid4()))
    
    logger.info(f"Processing job {job_id}")
    
    try:
        # Parse and validate input
        input_data = ImageEditInput(**event["input"])
        logger.info(f"Input validated: {input_data}")
        
        # Hash the image URL
        url_hash = sha256_hex(str(input_data.image_url))
        logger.info(f"URL hash: {url_hash}")
        
        # Timing variables
        download_start = time.time()
        download_time = 0
        infer_time = 0
        upload_time = 0
        
        # Check cache or download image
        cache_key = f"{S3_OBJECT_PREFIX}cache/{url_hash}"
        image_bytes = None
        extension = "png"
        content_type = "image/png"
        source = "cache"
        
        try:
            # Try to get from cache first
            logger.info(f"Checking cache for {cache_key}")
            response = minio_client.get_object(S3_BUCKET, cache_key + ".png")
            image_bytes = response.read()
            response.close()
            response.release_conn()
            extension = "png"
            content_type = "image/png"
        except S3Error as e:
            if e.code == "NoSuchKey":
                # Not in cache, download from URL
                logger.info("Image not in cache, downloading...")
                source = "download"
                download_start = time.time()
                image_bytes, extension, content_type = download_image(str(input_data.image_url))
                download_time = time.time() - download_start
                
                # Save to cache
                cache_key_with_ext = f"{cache_key}.{extension}"
                try:
                    minio_client.put_object(
                        S3_BUCKET,
                        cache_key_with_ext,
                        io.BytesIO(image_bytes),
                        len(image_bytes),
                        content_type=content_type
                    )
                    logger.info(f"Cached image as {cache_key_with_ext}")
                except Exception as e:
                    logger.warning(f"Failed to cache image: {e}")
            else:
                raise ValueError(f"Cache error: {e}")
        
        # Decode image
        image_stream = io.BytesIO(image_bytes)
        pil_image = Image.open(image_stream).convert("RGB")
        width, height = pil_image.size
        logger.info(f"Loaded image: {width}x{height} pixels")
        
        # Run image editing
        infer_start = time.time()
        # Prepare parameters for the model
        model_params = {
            "negative_prompt": input_data.negative_prompt,
            "seed": input_data.seed,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "strength": input_data.strength,
            "scheduler": input_data.scheduler,
            "safety_filter": input_data.safety_filter,
        }
        # Add extra parameters
        model_params.update(input_data.extra)
        
        edited_image = run_qwen_edit(
            model,
            pil_image,
            input_data.prompt,
            **model_params
        )
        infer_time = time.time() - infer_start
        
        # Encode result
        result_bytes, result_ext, result_content_type = encode_image(
            edited_image,
            input_data.output_format,
            input_data.output_quality
        )
        
        # Upload result
        upload_start = time.time()
        result_key = f"{S3_OBJECT_PREFIX}results/{url_hash}/{job_id}.{result_ext}"
        minio_client.put_object(
            S3_BUCKET,
            result_key,
            io.BytesIO(result_bytes),
            len(result_bytes),
            content_type=result_content_type
        )
        
        # Generate presigned URL
        presigned_url = minio_client.presigned_get_object(
            S3_BUCKET,
            result_key,
            expires=timedelta(seconds=PRESIGN_EXPIRY)
        )
        upload_time = time.time() - upload_start
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Prepare response metadata
        meta = {
            "source_url": str(input_data.image_url),
            "url_sha256": url_hash,
            "model": "Qwen/Qwen-Image-Edit",
            "prompt": input_data.prompt,
            "negative_prompt": input_data.negative_prompt,
            "seed": input_data.seed,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "strength": input_data.strength,
            "scheduler": input_data.scheduler,
            "runtime": {
                "latency_ms_total": int(total_time * 1000),
                "latency_ms_download": int(download_time * 1000),
                "latency_ms_infer": int(infer_time * 1000),
                "latency_ms_upload": int(upload_time * 1000),
            },
            "image": {
                "width": width,
                "height": height,
                "mode": pil_image.mode,
                "format": result_ext.upper()
            }
        }
        
        # Return success response
        return {
            "status": "success",
            "result": {
                "presigned_url": presigned_url,
                "bucket": S3_BUCKET,
                "object_key": result_key,
                "content_type": result_content_type,
                "expires_in": PRESIGN_EXPIRY,
                "meta": meta
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Return error response
        error_type = "ModelError"
        if isinstance(e, ValueError):
            error_type = "BadInput" if "Invalid" in str(e) else "DownloadFailed"
        elif isinstance(e, S3Error):
            error_type = "StorageError"
            
        return {
            "status": "error",
            "error": {
                "type": error_type,
                "message": str(e),
                "details": {
                    "job_id": job_id
                }
            }
        }

# Load model at cold start
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    raise

# Start the Runpod serverless handler
if __name__ == "__main__":
    logger.info("Starting Runpod serverless handler for Qwen-Image-Edit")
    runpod.serverless.start({"handler": handler})