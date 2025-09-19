# Start with a base image that has the full CUDA toolkit pre-installed.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables for non-interactive install and unbuffered Python
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies. We add the deadsnakes PPA to install Python 3.12.
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends git wget python3.12 python3-pip python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

# Set the working directory to ComfyUI
WORKDIR /ComfyUI

# Create a virtual environment with Python 3.12 and set the PATH to use its binaries
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in a single layer for efficiency
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir triton

# Set the CUDA architecture globally for the SageAttention build.
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install SageAttention from a pre-built wheel to avoid compilation issues.
RUN wget https://huggingface.co/nitin19/flash-attention-wheels/resolve/main/sageattention-2.1.1-cp312-cp312-linux_x86_64.whl && \
    pip install sageattention-2.1.1-cp312-cp312-linux_x86_64.whl && \
    rm sageattention-2.1.1-cp312-cp312-linux_x86_64.whl

# Unset the CUDA architecture variable.
ENV TORCH_CUDA_ARCH_LIST=""

# Clone the custom node repositories
RUN git clone https://github.com/kijai/ComfyUI-KJNodes/ custom_nodes/ComfyUI-KJNodes && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-KJNodes/requirements.txt
RUN git clone https://github.com/ClownsharkBatwing/RES4LYF/ custom_nodes/RES4LYF

# Create directories for models and custom nodes
RUN mkdir -p models/checkpoints models/loras models/vae models/diffusion_models/Wan models/diffusion_models/Qwen models/text_encoders/Qwen models/vae/Qwen custom_nodes

# --- Models are expected to be mounted via a Docker volume ---
# Create subfolders for LoRAs
RUN mkdir -p models/loras/Qwen models/loras/Wan

# --- Add your custom models and nodes here ---

# Download LoRAs for Qwen
RUN wget -O models/loras/Qwen/ https://civitai.com/api/download/models/2106185?type=Model&format=SafeTensor
RUN wget -O models/loras/Qwen/ https://civitai.com/api/download/models/1886273
RUN wget -O models/loras/Qwen/ https://civitai.com/api/download/models/2207719?type=Model&format=SafeTensor

# Download LoRA for Wan
RUN wget -O models/loras/Wan/ https://civitai.com/api/download/models/2066914?type=Model&format=SafeTensor

# --- Add InvokeAI installation ---
RUN pip install --no-cache-dir InvokeAI --use-pep517

# Expose the default ports for both ComfyUI and InvokeAI.
EXPOSE 8188 9090

# Set the entrypoint to run ComfyUI when the container starts.
# The PATH is set to the venv, so 'python' will point to the correct version.
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--use-pytorch-compile"]

