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
            # Prevent propagation to root logger to avoid duplicate logs
            self.logger.propagate = False
    
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
    scheduler: str = "EulerAncestral"
    extra: dict = {}

    @validator("image_url")
    def validate_image_url(cls, v):
        parsed = urlparse(str(v))
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("URL must be HTTP or HTTPS")
        return v

# Global model variable
model = None

def load_model():
    """Load the Qwen-Image-Edit model once at cold start with VRAM optimizations"""
    global model
    if model is None:
        job_id = "MODEL_INIT"
        logger.info(job_id, "Loading Qwen-Image-Edit model with VRAM optimizations...")
        load_start = time.time()
        
        try:
            # Import required modules
            from diffusers import QwenImageEditPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import load_image
            import torch
            
            # Log CUDA availability
            if not torch.cuda.is_available():
                logger.error(job_id, "CUDA is not available. This application requires a GPU.")
                raise RuntimeError("CUDA is not available. This application requires a GPU.")
            
            logger.info(job_id, "CUDA is available", device_count=torch.cuda.device_count())
            
            # Load the Qwen model with memory optimizations
            model_load_start = time.time()
            
            # Load model with optimizations
            model_load_start = time.time()
            
            # Load model with optimizations
            logger.info(job_id, "Loading QwenImageEditPipeline from_pretrained")
            model = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for lower memory usage
                token=HF_TOKEN,
            )
            logger.info(job_id, "QwenImageEditPipeline loaded successfully")
            
            # Move model to GPU
            move_start = time.time()
            model = model.to("cuda")
            move_time = time.time() - move_start
            logger.info(job_id, "Moved model to CUDA", move_time=f"{move_time:.2f}s")
            
            # Enable attention slicing for additional memory savings
            try:
                model.enable_attention_slicing()
                logger.info(job_id, "Enabled attention slicing")
            except Exception as e:
                logger.warning(job_id, "Failed to enable attention slicing", error=str(e))
            
            # Enable VAE slicing for additional memory savings
            try:
                model.enable_vae_slicing()
                logger.info(job_id, "Enabled VAE slicing")
            except Exception as e:
                logger.warning(job_id, "Failed to enable VAE slicing", error=str(e))
            
            # Enable VAE tiling for processing larger images with limited memory
            try:
                model.enable_vae_tiling()
                logger.info(job_id, "Enabled VAE tiling")
            except Exception as e:
                logger.warning(job_id, "Failed to enable VAE tiling", error=str(e))
            
            model_load_time = time.time() - model_load_start
            logger.info(job_id, "Successfully loaded Qwen/Qwen-Image-Edit model", 
                       load_time=f"{model_load_time:.2f}s",
                       dtype=str(model.dtype))
            
            load_time = time.time() - load_start
            logger.info(job_id, "Model loaded successfully", total_load_time=f"{load_time:.2f}s")
        except Exception as e:
            logger.error(job_id, "Failed to load model", error=str(e))
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
    """Run Qwen-Image-Edit on the input image with VRAM optimizations"""
    logger.info(job_id, "Running Qwen-Image-Edit", prompt=prompt)
    logger.debug(job_id, "Model parameters", **kwargs)
    
    try:
        
        # Extract parameters
        negative_prompt = kwargs.get("negative_prompt", "")
        seed = kwargs.get("seed", None)
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        true_cfg_scale = kwargs.get("true_cfg_scale", 1.0)  # Use true_cfg_scale for classifier-free guidance
        guidance_scale = kwargs.get("guidance_scale", 1.0)  # This is a different parameter
        scheduler = kwargs.get("scheduler", "EulerAncestral")
        
        # Limit inference steps to reduce memory usage
        num_inference_steps = min(num_inference_steps, 50)  # Cap at 50 steps
        
        # Validate true_cfg_scale parameter
        if true_cfg_scale < 1.0:
            logger.warning(job_id, "true_cfg_scale should be >= 1.0, setting to default value", true_cfg_scale=true_cfg_scale)
            true_cfg_scale = 1.0
        elif true_cfg_scale > 10.0:
            logger.warning(job_id, "true_cfg_scale is very high, this may cause issues", true_cfg_scale=true_cfg_scale)
            
        # Validate guidance_scale parameter
        if guidance_scale < 1.0:
            logger.warning(job_id, "guidance_scale should be >= 1.0, setting to default value", guidance_scale=guidance_scale)
            guidance_scale = 1.0
        elif guidance_scale > 20.0:
            logger.warning(job_id, "guidance_scale is very high, this may cause issues", guidance_scale=guidance_scale)
        
        # Set scheduler if specified
        scheduler_start = time.time()
        try:
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
        except Exception as scheduler_error:
            logger.error(job_id, "Error configuring scheduler", scheduler=scheduler, error=str(scheduler_error))
            # Continue with default scheduler if there's an error
            pass
        
        # Set generator for reproducible results if seed is provided
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=model.device).manual_seed(seed)
            logger.debug(job_id, "Generator set with seed", seed=seed)
        
        # Run the model
        infer_start = time.time()
        try:
            logger.debug(job_id, "Calling model with parameters", 
                        prompt=prompt, 
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        true_cfg_scale=true_cfg_scale,
                        guidance_scale=guidance_scale,
                        generator_type=type(generator).__name__ if generator else "None",
                        return_dict=True,  # Use return_dict=True to get QwenImagePipelineOutput
                        image_type=type(image).__name__,
                        image_mode=getattr(image, 'mode', 'N/A') if image else 'N/A')
            
            # Prepare the model call parameters
            model_kwargs = {
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,  # Use true_cfg_scale for classifier-free guidance
                "guidance_scale": guidance_scale,  # Keep guidance_scale as a separate parameter
                "generator": generator,
                "return_dict": True,  # Use return_dict=True for consistent handling
            }
            
            result = model(**model_kwargs)
            
            logger.debug(job_id, "Model call completed successfully")
        except Exception as model_error:
            logger.error(job_id, "Error calling model", error=str(model_error), exc_info=True)
            raise ValueError(f"Failed to call model: {str(model_error)}")
        
        infer_time = time.time() - infer_start
        
        # Clear GPU cache after inference to free up memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(job_id, "Cleared CUDA cache after inference")
        except Exception as e:
            logger.warning(job_id, "Failed to clear CUDA cache", error=str(e))
        
        # Handle the result - the model should return a QwenImagePipelineOutput when return_dict=True
        logger.debug(job_id, "Model result type", result_type=type(result).__name__)
        
        # According to the documentation, when return_dict=True, we get a QwenImagePipelineOutput
        if hasattr(result, 'images') and result.images:
            logger.info(job_id, "Model inference completed successfully", 
                       inference_time=f"{infer_time:.2f}s",
                       result_type="QwenImagePipelineOutput",
                       num_images=len(result.images))
            # Get the first image from the list
            result_image = result.images[0]
            
            # Log information about the result image before processing
            logger.debug(job_id, "Raw result image info", 
                        mode=getattr(result_image, 'mode', 'N/A'),
                        size=getattr(result_image, 'size', 'N/A') if hasattr(result_image, 'size') else 'N/A')
            
            return result_image
        else:
            raise ValueError(f"Model returned unexpected result: {type(result)}")
            
    except Exception as e:
        logger.error(job_id, "Error during model inference", error=str(e))
        # Clear GPU cache even on error
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(job_id, "Cleared CUDA cache after inference error")
        except Exception as ce:
            logger.warning(job_id, "Failed to clear CUDA cache after error", error=str(ce))
        # Raise the error instead of returning the original image
        raise


def handler(event):
    """Main handler function for Runpod serverless"""
    start_time = time.time()
    job_id = event.get("id", str(uuid.uuid4()))
    
    logger.info(job_id, "Processing job")
    
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
        
        # Download image directly using diffusers.utils.load_image
        logger.info(job_id, "Downloading image directly from URL")
        download_start = time.time()
        try:
            pil_image = load_image(str(input_data.image_url)).convert("RGB")
            width, height = pil_image.size
            download_time = time.time() - download_start
            logger.info(job_id, "Image downloaded successfully", 
                       width=width, 
                       height=height, 
                       mode=pil_image.mode,
                       download_time=f"{download_time:.2f}s")
            
            
        except Exception as download_error:
            logger.error(job_id, "Error downloading image", error=str(download_error))
            raise ValueError(f"Failed to download image: {str(download_error)}")
        
        # Run image editing
        logger.info(job_id, "Starting image editing")
        # Prepare parameters for the model
        model_params = {
            "negative_prompt": input_data.negative_prompt,
            "num_inference_steps": input_data.num_inference_steps,
            "true_cfg_scale": 1.0,  # Default value for true_cfg_scale
            "guidance_scale": input_data.guidance_scale,
            "scheduler": input_data.scheduler,
        }
        
        # Handle seed parameter - generate random seed if not provided or invalid
        if input_data.seed is None or input_data.seed <= 0:
            import random
            model_params["seed"] = random.randint(1, 2**32 - 1)
            logger.info(job_id, "Generated random seed", seed=model_params["seed"])
        else:
            model_params["seed"] = input_data.seed
            
        # Add extra parameters (this allows users to override true_cfg_scale and other parameters)
        model_params.update(input_data.extra)
        
        # Validate extra parameters
        if "true_cfg_scale" in model_params:
            true_cfg_scale = model_params["true_cfg_scale"]
            if true_cfg_scale < 1.0:
                logger.warning(job_id, "true_cfg_scale in extra params should be >= 1.0, setting to default value", true_cfg_scale=true_cfg_scale)
                model_params["true_cfg_scale"] = 1.0
            elif true_cfg_scale > 10.0:
                logger.warning(job_id, "true_cfg_scale in extra params is very high, this may cause issues", true_cfg_scale=true_cfg_scale)
                
        if "guidance_scale" in model_params:
            guidance_scale = model_params["guidance_scale"]
            if guidance_scale < 1.0:
                logger.warning(job_id, "guidance_scale in extra params should be >= 1.0, setting to default value", guidance_scale=guidance_scale)
                model_params["guidance_scale"] = 1.0
            elif guidance_scale > 20.0:
                logger.warning(job_id, "guidance_scale in extra params is very high, this may cause issues", guidance_scale=guidance_scale)
        
        # Log all model parameters for debugging
        logger.debug(job_id, "Model parameters", **model_params)
        
        try:
            edited_image = run_qwen_edit(
                job_id,
                model,
                pil_image,
                input_data.prompt,
                **model_params
            )
        except Exception as e:
            logger.error(job_id, "Error during image editing", error=str(e))
            raise ValueError(f"Failed to edit image: {str(e)}")
        
        # Save result as PNG temporarily
        try:
            result_filename = f"{url_hash}.png"
            edited_image.save(result_filename)
            
            # Read the saved PNG file
            with open(result_filename, "rb") as f:
                result_bytes = f.read()
            
            result_ext = "png"
            result_content_type = "image/png"
        except Exception as save_error:
            logger.error(job_id, "Error saving result image", error=str(save_error))
            raise ValueError(f"Failed to save result image: {str(save_error)}")
        
        # Upload result
        upload_start = time.time()
        try:
            result_key = f"{S3_OBJECT_PREFIX}results/{url_hash}/{job_id}.{result_ext}"
            minio_client.put_object(
                S3_BUCKET,
                result_key,
                io.BytesIO(result_bytes),
                len(result_bytes),
                content_type=result_content_type
            )
            
            # Remove the temporary file
            try:
                os.remove(result_filename)
                logger.debug(job_id, "Temporary file removed", filename=result_filename)
            except Exception as remove_error:
                logger.warning(job_id, "Failed to remove temporary file", error=str(remove_error))
            
            # Generate presigned URL
            presigned_url = minio_client.presigned_get_object(
                S3_BUCKET,
                result_key,
                expires=timedelta(seconds=PRESIGN_EXPIRY)
            )
            upload_time = time.time() - upload_start
        except Exception as upload_error:
            logger.error(job_id, "Error uploading result", error=str(upload_error))
            raise ValueError(f"Failed to upload result: {str(upload_error)}")
        
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
            "seed": model_params["seed"],  # Use the actual seed used in generation
            "num_inference_steps": input_data.num_inference_steps,
            "true_cfg_scale": model_params.get("true_cfg_scale", 1.0),
            "guidance_scale": input_data.guidance_scale,
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
    logger.info("MODEL_INIT", "Starting model loading at cold start")
    load_model()
    logger.info("MODEL_INIT", "Model loaded successfully at cold start")
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