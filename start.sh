#!/bin/bash
# Start script for LivePortrait RunPod Serverless

echo "Starting ComfyUI server in background..."
cd /content/ComfyUI
python main.py --listen --port 7860 &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
MAX_WAIT=120
WAITED=0
while ! curl -s http://127.0.0.1:7860/system_stats > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ComfyUI failed to start within $MAX_WAIT seconds"
        exit 1
    fi
    echo "Waiting... ($WAITED/$MAX_WAIT seconds)"
done

echo "ComfyUI is ready!"

# Start the RunPod handler
echo "Starting RunPod handler..."
cd /content
python handler.py
