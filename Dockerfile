# LivePortrait RunPod Serverless Worker
# Based on camenduru/liveportrait-runpod with updated ComfyUI dependencies
# Updated: 2024-12-24

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /content

# System packages
RUN apt-get update && apt-get install -y \
    git git-lfs aria2 ffmpeg unzip wget \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python ML packages
RUN pip install --upgrade pip && \
    pip install xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 \
    diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 \
    opencv-python==4.9.0.80 imageio==2.34.1 imageio-ffmpeg==0.4.9 \
    onnxruntime-gpu==1.18.0 mediapipe==0.10.14 insightface==0.7.3

# Install ComfyUI's new required packages (added in late 2024)
RUN pip install alembic aiohttp aiosqlite \
    comfyui-workflow-templates comfyui-embedded-docs \
    || pip install alembic aiohttp aiosqlite

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI

# Install ComfyUI requirements
RUN cd /content/ComfyUI && \
    pip install -r requirements.txt || true

# Clone ComfyUI custom nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ /content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/cubiq/ComfyUI_essentials /content/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts /content/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux /content/ComfyUI/custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui /content/ComfyUI/custom_nodes/was-node-suite-comfyui

# Install custom node requirements
RUN cd /content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ && \
    pip install -r requirements.txt || true
RUN cd /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt || true
RUN cd /content/ComfyUI/custom_nodes/comfyui_controlnet_aux && \
    pip install -r requirements.txt || true
RUN cd /content/ComfyUI/custom_nodes/was-node-suite-comfyui && \
    pip install -r requirements.txt || true

# Download InsightFace models
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/insightface/models/buffalo_l/det_10g.onnx \
    -d /content/ComfyUI/models/insightface/models/buffalo_l -o det_10g.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/insightface/models/buffalo_l/2d106det.onnx \
    -d /content/ComfyUI/models/insightface/models/buffalo_l -o 2d106det.onnx

# Download LivePortrait models
RUN mkdir -p /content/ComfyUI/models/liveportrait && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.safetensors \
    -d /content/ComfyUI/models/liveportrait/base_models -o appearance_feature_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/landmark.onnx \
    -d /content/ComfyUI/models/liveportrait/base_models -o landmark.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.safetensors \
    -d /content/ComfyUI/models/liveportrait/base_models -o motion_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.safetensors \
    -d /content/ComfyUI/models/liveportrait/base_models -o spade_generator.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/stitching_retargeting_module.safetensors \
    -d /content/ComfyUI/models/liveportrait/base_models -o stitching_retargeting_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/base_models/warping_module.safetensors \
    -d /content/ComfyUI/models/liveportrait/base_models -o warping_module.safetensors

# Download retargeting models
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.safetensors \
    -d /content/ComfyUI/models/liveportrait/retargeting_models -o stitching_retargeting_module.safetensors

# Download animal models (optional but included)
RUN mkdir -p /content/ComfyUI/models/liveportrait_animals/base_models && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/base_models/appearance_feature_extractor.safetensors \
    -d /content/ComfyUI/models/liveportrait_animals/base_models -o appearance_feature_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/base_models/motion_extractor.safetensors \
    -d /content/ComfyUI/models/liveportrait_animals/base_models -o motion_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/base_models/spade_generator.safetensors \
    -d /content/ComfyUI/models/liveportrait_animals/base_models -o spade_generator.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/base_models/warping_module.safetensors \
    -d /content/ComfyUI/models/liveportrait_animals/base_models -o warping_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/base_models/xpose.pth \
    -d /content/ComfyUI/models/liveportrait_animals/base_models -o xpose.pth

# Download retargeting animal models
RUN mkdir -p /content/ComfyUI/models/liveportrait_animals/retargeting_models && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/camenduru/LivePortrait/resolve/main/liveportrait_animals/retargeting_models/stitching_retargeting_module.safetensors \
    -d /content/ComfyUI/models/liveportrait_animals/retargeting_models -o stitching_retargeting_module.safetensors

WORKDIR /content/ComfyUI

# Install RunPod SDK
RUN pip install runpod requests

# Copy handler
COPY handler.py /content/handler.py
COPY start.sh /content/start.sh
RUN chmod +x /content/start.sh

# Start handler (which starts ComfyUI in background)
CMD ["/content/start.sh"]
