"""
RunPod Serverless Handler for LivePortrait

This handler receives requests from RunPod's serverless API and processes them
through ComfyUI with LivePortrait nodes.

Input format:
{
    "input": {
        "source_image": "base64 encoded image or URL",
        "driving_video": "base64 encoded video or URL",
        "audio": "optional base64 audio or URL for lip sync",
        "flag_relative": true,
        "flag_do_crop": true,
        "flag_pasteback": true
    }
}

Output format:
{
    "video_url": "URL to the generated video" or
    "video_base64": "base64 encoded video"
}
"""

import os
import sys
import json
import base64
import tempfile
import subprocess
import time
import urllib.request
from pathlib import Path

# Add ComfyUI to path
sys.path.insert(0, '/content/ComfyUI')

import runpod

# Configuration
COMFYUI_DIR = "/content/ComfyUI"
INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
OUTPUT_DIR = os.path.join(COMFYUI_DIR, "output")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL to the specified path."""
    print(f"Downloading {url} to {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def save_base64_file(data: str, output_path: str) -> str:
    """Save base64 encoded data to a file."""
    # Remove data URI prefix if present
    if ',' in data:
        data = data.split(',', 1)[1]

    binary_data = base64.b64decode(data)
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    return output_path


def file_to_base64(file_path: str) -> str:
    """Convert a file to base64 string."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_liveportrait_workflow(source_image_path: str, driving_video_path: str,
                               audio_path: str = None, **kwargs) -> dict:
    """
    Generate a ComfyUI workflow for LivePortrait.

    Workflow structure:
    1. DownloadAndLoadLivePortraitModels -> pipeline
    2. LivePortraitLoadCropper -> cropper
    3. LoadImage -> source_image
    4. VHS_LoadVideo -> driving_images (frames)
    5. LivePortraitCropper -> crop_info
    6. LivePortraitProcess -> animated frames
    7. VHS_VideoCombine -> output video
    """
    workflow = {
        # Node 1: Load LivePortrait pipeline/models
        "1": {
            "class_type": "DownloadAndLoadLivePortraitModels",
            "inputs": {
                "precision": "auto",
                "mode": "human"
            }
        },
        # Node 2: Load face cropper (InsightFace-based)
        "2": {
            "class_type": "LivePortraitLoadCropper",
            "inputs": {
                "onnx_device": "CUDA",
                "keep_model_loaded": True,
                "detection_threshold": 0.5
            }
        },
        # Node 3: Load source image
        "3": {
            "class_type": "LoadImage",
            "inputs": {
                "image": os.path.basename(source_image_path)
            }
        },
        # Node 4: Load driving video as frames
        "4": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": os.path.basename(driving_video_path),
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1
            }
        },
        # Node 5: Crop and prepare source face
        "5": {
            "class_type": "LivePortraitCropper",
            "inputs": {
                "pipeline": ["1", 0],
                "cropper": ["2", 0],
                "source_image": ["3", 0],
                "dsize": 512,
                "scale": 2.3,
                "vx_ratio": 0.0,
                "vy_ratio": -0.125,
                "face_index": 0,
                "face_index_order": "large-small",
                "rotate": True
            }
        },
        # Node 6: Process with LivePortrait
        "6": {
            "class_type": "LivePortraitProcess",
            "inputs": {
                "pipeline": ["1", 0],
                "crop_info": ["5", 1],  # Index 1 is CROPINFO, index 0 is cropped IMAGE
                "source_image": ["3", 0],
                "driving_images": ["4", 0],
                "lip_zero": False,
                "lip_zero_threshold": 0.03,
                "stitching": True,
                "delta_multiplier": kwargs.get("driving_multiplier", 1.0),
                "mismatch_method": "constant",
                "relative_motion_mode": kwargs.get("relative_motion_mode", "relative"),
                "driving_smooth_observation_variance": 3e-6,
                "expression_friendly": False,
                "expression_friendly_multiplier": 1.0
            }
        },
        # Node 7: Combine frames into video
        "7": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["6", 0],
                "frame_rate": 25,
                "loop_count": 0,
                "filename_prefix": "liveportrait_output",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True
            }
        }
    }

    return workflow


def run_comfyui_workflow(workflow: dict) -> dict:
    """
    Execute a ComfyUI workflow and return the output.

    This uses ComfyUI's API to queue and execute the workflow.
    """
    import requests

    # ComfyUI API endpoint (running locally in the container)
    api_url = "http://127.0.0.1:7860"

    # Queue the prompt
    response = requests.post(
        f"{api_url}/prompt",
        json={"prompt": workflow}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to queue prompt: {response.text}")

    result = response.json()
    prompt_id = result.get("prompt_id")

    if not prompt_id:
        raise Exception(f"No prompt_id in response: {result}")

    # Poll for completion
    max_wait = 300  # 5 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        history_response = requests.get(f"{api_url}/history/{prompt_id}")

        if history_response.status_code == 200:
            history = history_response.json()
            if prompt_id in history:
                return history[prompt_id]

        time.sleep(2)

    raise Exception("Workflow execution timed out")


def handler(event: dict) -> dict:
    """
    RunPod serverless handler function.

    Receives input, processes through LivePortrait, returns output.
    """
    try:
        job_input = event.get("input", {})

        print(f"Received job input: {list(job_input.keys())}")

        # Process source image
        source_image = job_input.get("source_image") or job_input.get("source_image_url")
        if not source_image:
            return {"error": "source_image or source_image_url is required"}

        source_path = os.path.join(INPUT_DIR, "source.png")
        if source_image.startswith("http"):
            download_file(source_image, source_path)
        else:
            save_base64_file(source_image, source_path)

        # Process driving video
        driving_video = job_input.get("driving_video") or job_input.get("driving_video_url")
        if not driving_video:
            return {"error": "driving_video or driving_video_url is required"}

        driving_path = os.path.join(INPUT_DIR, "driving.mp4")
        if driving_video.startswith("http"):
            download_file(driving_video, driving_path)
        else:
            save_base64_file(driving_video, driving_path)

        # Process optional audio
        audio_path = None
        audio = job_input.get("audio") or job_input.get("audio_url")
        if audio:
            audio_path = os.path.join(INPUT_DIR, "audio.wav")
            if audio.startswith("http"):
                download_file(audio, audio_path)
            else:
                save_base64_file(audio, audio_path)

        print(f"Source image saved to: {source_path}")
        print(f"Driving video saved to: {driving_path}")
        if audio_path:
            print(f"Audio saved to: {audio_path}")

        # Generate workflow
        workflow = get_liveportrait_workflow(
            source_image_path=source_path,
            driving_video_path=driving_path,
            audio_path=audio_path,
            flag_relative=job_input.get("flag_relative", True),
            flag_do_crop=job_input.get("flag_do_crop", True),
            flag_pasteback=job_input.get("flag_pasteback", True),
            driving_smooth=job_input.get("driving_smooth", True),
            driving_multiplier=job_input.get("driving_multiplier", 1.0)
        )

        print("Generated workflow, executing...")

        # Execute workflow
        result = run_comfyui_workflow(workflow)

        print(f"Workflow completed: {result}")

        # Debug: List all files in output directory
        print(f"Checking output directory: {OUTPUT_DIR}")
        all_files = list(Path(OUTPUT_DIR).glob("*"))
        print(f"All files in output: {[str(f) for f in all_files]}")

        # Find output video - check multiple patterns
        output_files = list(Path(OUTPUT_DIR).glob("liveportrait_output*.mp4"))
        if not output_files:
            # Try any mp4 file
            output_files = list(Path(OUTPUT_DIR).glob("*.mp4"))
        if not output_files:
            # Try any video file
            output_files = list(Path(OUTPUT_DIR).glob("*.*"))
            output_files = [f for f in output_files if f.suffix.lower() in ['.mp4', '.webm', '.avi', '.mov']]

        if not output_files:
            return {"error": f"No output video generated. Files in output dir: {[str(f) for f in all_files]}"}

        # Get the most recent output
        output_file = max(output_files, key=os.path.getctime)

        # Return as base64
        video_base64 = file_to_base64(str(output_file))

        return {
            "video_base64": f"data:video/mp4;base64,{video_base64}"
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
