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

# Configure logging with custom formatter for structured logging
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        
        # Avoid adding multiple handlers if the logger already has handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _format_message(self, job_id: str, message: str, **kwargs) -> str:
        """Format log message with job ID and additional context"""
        base_msg = f"[Job: {job_id}] {message}"
        if kwargs:
            context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            return f"{base_msg} | {context}"
        return base_msg
    
    def info(self, job_id: str, message: str, **kwargs):
        self.logger.info(self._format_message(job_id, message, **kwargs))
        
    def warning(self, job_id: str, message: str, **kwargs):
        self.logger.warning(self._format_message(job_id, message, **kwargs))
        
    def error(self, job_id: str, message: str, **kwargs):
        self.logger.error(self._format_message(job_id, message, **kwargs))
        
    def debug(self, job_id: str, message: str, **kwargs):
        self.logger.debug(self._format_message(job_id, message, **kwargs))

logger = StructuredLogger(__name__)

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
        job_id = "MODEL_INIT"
        logger.info(job_id, "Loading Qwen-Image-Edit model...")
        load_start = time.time()
        
        try:
            # Import required modules
            from diffusers import QwenImagePipeline, DPMSolverMultistepScheduler
            import torch
            
            # Log CUDA availability
            if torch.cuda.is_available():
                logger.info(job_id, "CUDA is available", device_count=torch.cuda.device_count())
            else:
                logger.info(job_id, "CUDA is not available, using CPU")
            
            # Try to load the Qwen model first
            try:
                model_load_start = time.time()
                model = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    token=HF_TOKEN,
                )
                model_load_time = time.time() - model_load_start
                logger.info(job_id, "Successfully loaded Qwen/Qwen-Image-Edit model", 
                           load_time=f"{model_load_time:.2f}s",
                           dtype=str(model.dtype))
            except Exception as e:
                logger.warning(job_id, "Failed to load Qwen/Qwen-Image-Edit model", error=str(e))
                logger.info(job_id, "Falling back to timbrooks/instruct-pix2pix model")
                # Fallback to the original InstructPix2Pix model
                model_load_start = time.time()
                model = QwenImagePipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    token=HF_TOKEN,
                )
                model_load_time = time.time() - model_load_start
                logger.info(job_id, "Loaded timbrooks/instruct-pix2pix model", 
                           load_time=f"{model_load_time:.2f}s",
                           dtype=str(model.dtype))
            
            # Move to GPU if available
            if torch.cuda.is_available():
                move_start = time.time()
                model = model.to("cuda")
                move_time = time.time() - move_start
                logger.info(job_id, "Moved model to CUDA", move_time=f"{move_time:.2f}s")
            
            # Use DPMSolverMultistepScheduler for better results
            scheduler_start = time.time()
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
            scheduler_time = time.time() - scheduler_start
            logger.info(job_id, "Scheduler configured", 
                       scheduler_type="DPMSolverMultistep",
                       config_time=f"{scheduler_time:.2f}s")
            
            model.eval()
            load_time = time.time() - load_start
            logger.info(job_id, "Model loaded successfully", total_load_time=f"{load_time:.2f}s")
        except Exception as e:
            logger.error(job_id, "Failed to load model", error=str(e))
            # Fallback to CPU if CUDA is not available
            try:
                from diffusers import QwenImagePipeline
                import torch
                
                # Try to load the Qwen model first
                try:
                    model_load_start = time.time()
                    model = QwenImagePipeline.from_pretrained(
                        "Qwen/Qwen-Image-Edit",
                        torch_dtype=torch.float32,
                        token=HF_TOKEN,
                    )
                    model_load_time = time.time() - model_load_start
                    logger.info(job_id, "Successfully loaded Qwen/Qwen-Image-Edit model on CPU", 
                               load_time=f"{model_load_time:.2f}s")
                except Exception as e:
                    logger.warning(job_id, "Failed to load Qwen/Qwen-Image-Edit model on CPU", error=str(e))
                    logger.info(job_id, "Falling back to timbrooks/instruct-pix2pix model on CPU")
                    # Fallback to the original InstructPix2Pix model
                    model_load_start = time.time()
                    model = QwenImagePipeline.from_pretrained(
                        "timbrooks/instruct-pix2pix",
                        torch_dtype=torch.float32,
                        token=HF_TOKEN,
                    )
                    model_load_time = time.time() - model_load_start
                    logger.info(job_id, "Loaded timbrooks/instruct-pix2pix model on CPU", 
                               load_time=f"{model_load_time:.2f}s")
                
                scheduler_start = time.time()
                model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
                scheduler_time = time.time() - scheduler_start
                logger.info(job_id, "Scheduler configured on CPU", 
                           scheduler_type="DPMSolverMultistep",
                           config_time=f"{scheduler_time:.2f}s")
                
                model.eval()
                load_time = time.time() - load_start
                logger.info(job_id, "Model loaded on CPU successfully", total_load_time=f"{load_time:.2f}s")
            except Exception as e2:
                logger.error(job_id, "Failed to load model on CPU", error=str(e2))
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

def download_image(job_id: str, url: str) -> Tuple[bytes, str, str]:
    """Download image from URL with validation"""
    logger.info(job_id, "Starting image download", url=url)
    download_start = time.time()
    
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
        
        download_time = time.time() - download_start
        logger.info(job_id, "Image downloaded successfully", 
                   bytes=len(image_bytes), 
                   content_type=content_type, 
                   extension=extension,
                   download_time=f"{download_time:.2f}s")
        return image_bytes, extension, content_type
        
    except requests.exceptions.RequestException as e:
        logger.error(job_id, "Failed to download image", error=str(e))
        raise ValueError(f"Failed to download image: {str(e)}")

def encode_image(job_id: str, image: PILImage, format: str, quality: int = 95) -> Tuple[bytes, str, str]:
    """Encode PIL image to bytes with specified format"""
    logger.debug(job_id, "Encoding image", format=format, quality=quality, mode=image.mode)
    encode_start = time.time()
    
    buffer = io.BytesIO()
    
    if format == "jpeg":
        # Convert RGBA to RGB if needed for JPEG
        if image.mode in ("RGBA", "LA", "P"):
            logger.debug(job_id, "Converting image mode for JPEG", from_mode=image.mode, to_mode="RGB")
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
    
    encode_time = time.time() - encode_start
    logger.info(job_id, "Image encoded successfully", 
               bytes=len(image_bytes), 
               format=format,
               encode_time=f"{encode_time:.2f}s")
    return image_bytes, extension, content_type

def run_qwen_edit(job_id: str, model, image: PILImage, prompt: str, **kwargs) -> PILImage:
    """Run Qwen-Image-Edit on the input image"""
    logger.info(job_id, "Running Qwen-Image-Edit", prompt=prompt)
    logger.debug(job_id, "Model parameters", **kwargs)
    
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
        scheduler_start = time.time()
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
        
        scheduler_time = time.time() - scheduler_start
        logger.debug(job_id, "Scheduler configured", scheduler=scheduler, config_time=f"{scheduler_time:.2f}s")
        
        # Set generator for reproducible results if seed is provided
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=model.device).manual_seed(seed)
            logger.debug(job_id, "Generator set with seed", seed=seed)
        
        # Run the model
        infer_start = time.time()
        result = model(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        )
        infer_time = time.time() - infer_start
        
        # Return the edited image
        if hasattr(result, 'images') and result.images:
            logger.info(job_id, "Model inference completed successfully", 
                       inference_time=f"{infer_time:.2f}s",
                       result_type="images_object")
            return result.images[0]
        elif isinstance(result, list) and len(result) > 0:
            logger.info(job_id, "Model inference completed successfully", 
                       inference_time=f"{infer_time:.2f}s",
                       result_type="list")
            return result[0]
        else:
            # Fallback to original image if model didn't return an image
            logger.warning(job_id, "Model didn't return an edited image, returning original")
            return image
            
    except Exception as e:
        logger.error(job_id, "Error during model inference", error=str(e))
        # Return original image if there's an error
        return image


def warmup_model(job_id: str) -> dict:
    """Warm up the model by running a simple inference"""
    logger.info(job_id, "Starting model warmup")
    warmup_start = time.time()
    
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Create a simple test image (black 64x64 image)
        import numpy as np
        test_image_array = np.zeros((64, 64, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image_array)
        
        # Run a simple inference with minimal steps
        warmup_params = {
            "num_inference_steps": 1,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "strength": 0.8
        }
        
        logger.info(job_id, "Running warmup inference", **warmup_params)
        result_image = run_qwen_edit(
            job_id,
            model,
            test_image,
            "warmup test",
            **warmup_params
        )
        
        warmup_time = time.time() - warmup_start
        logger.info(job_id, "Model warmup completed successfully", warmup_time=f"{warmup_time:.2f}s")
        
        return {
            "status": "success",
            "result": {
                "message": "Model warmed up successfully",
                "warmup_time": warmup_time
            }
        }
    except Exception as e:
        logger.error(job_id, "Error during model warmup", error=str(e), exc_info=True)
        return {
            "status": "error",
            "error": {
                "type": "WarmupError",
                "message": str(e),
                "details": {
                    "job_id": job_id
                }
            }
        }

def handler(event):
    """Main handler function for Runpod serverless"""
    start_time = time.time()
    job_id = event.get("id", str(uuid.uuid4()))
    
    logger.info(job_id, "Processing job")
    
    # Check if this is a warmup request
    if event.get("input", {}).get("warmup", False):
        logger.info(job_id, "Processing warmup request")
        return warmup_model(job_id)
    
    try:
        # Parse and validate input
        input_data = ImageEditInput(**event["input"])
        logger.info(job_id, "Input validated", 
                   prompt=input_data.prompt,
                   image_url=str(input_data.image_url),
                   num_inference_steps=input_data.num_inference_steps,
                   guidance_scale=input_data.guidance_scale)
        
        # Hash the image URL
        url_hash = sha256_hex(str(input_data.image_url))
        logger.debug(job_id, "URL hashed", url_hash=url_hash)
        
        # Timing variables
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
            logger.info(job_id, "Checking cache", cache_key=cache_key)
            response = minio_client.get_object(S3_BUCKET, cache_key + ".png")
            image_bytes = response.read()
            response.close()
            response.release_conn()
            extension = "png"
            content_type = "image/png"
            logger.info(job_id, "Image found in cache")
        except S3Error as e:
            if e.code == "NoSuchKey":
                # Not in cache, download from URL
                logger.info(job_id, "Image not in cache, downloading...")
                source = "download"
                download_start = time.time()
                image_bytes, extension, content_type = download_image(job_id, str(input_data.image_url))
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
                    logger.info(job_id, "Cached image", cache_key=cache_key_with_ext)
                except Exception as e:
                    logger.warning(job_id, "Failed to cache image", error=str(e))
            else:
                logger.error(job_id, "Cache error", error=str(e))
                raise ValueError(f"Cache error: {e}")
        
        # Decode image
        decode_start = time.time()
        image_stream = io.BytesIO(image_bytes)
        pil_image = Image.open(image_stream).convert("RGB")
        width, height = pil_image.size
        decode_time = time.time() - decode_start
        logger.info(job_id, "Image loaded", 
                   width=width, 
                   height=height, 
                   mode=pil_image.mode,
                   decode_time=f"{decode_time:.2f}s")
        
        # Run image editing
        logger.info(job_id, "Starting image editing")
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
            job_id,
            model,
            pil_image,
            input_data.prompt,
            **model_params
        )
        
        # Encode result
        result_bytes, result_ext, result_content_type = encode_image(
            job_id,
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
        
        logger.info(job_id, "Job completed successfully", 
                   total_time=f"{total_time:.2f}s",
                   source=source)
        
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
        logger.error(job_id, "Error processing job", error=str(e), exc_info=True)
        
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
    # Create a temporary logger instance for this error
    temp_logger = StructuredLogger(__name__)
    temp_logger.error("MODEL_INIT", "Failed to load model at startup", error=str(e))
    raise

# Start the Runpod serverless handler
if __name__ == "__main__":
    # Create a temporary logger instance for startup
    temp_logger = StructuredLogger(__name__)
    temp_logger.info("STARTUP", "Starting Runpod serverless handler for Qwen-Image-Edit")
    runpod.serverless.start({"handler": handler})