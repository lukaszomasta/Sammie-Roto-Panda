#!/usr/bin/env bash

# Always operate relative to this script's folder
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to cd to script directory"; exit 1; }

# Function to check Python version
check_python_version() {
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 is not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.10"

    version_ge() {
        [[ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
    }

    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
        echo "ERROR: Python 3.10 or higher is required. Detected version: $PYTHON_VERSION"
        exit 1
    fi
	
	# Check if venv is available
	if ! python3 -m venv --help &> /dev/null; then
		echo "ERROR: The 'venv' module is not available in this Python installation."
		echo "Try reinstalling Python from https://www.python.org/ or via your package manager."
		exit 1
	fi
	
	echo "Detected compatible Python version: $PYTHON_VERSION"
}

# Check Python version
check_python_version

# Detect OS
OS_TYPE="$(uname)"

# Create and activate virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation..."
else
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip and install wheel
echo "Upgrading pip..."
pip3 install --upgrade pip
echo "Installing wheel..."
pip3 install wheel

# Install PyTorch
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Detected macOS. Installing PyTorch..."
    pip3 install torch==2.11.0 torchvision torchaudio
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Detected Linux."
	echo
    echo "Which PyTorch build do you want to install?"
    select PT_VERSION in "CUDA 13.0 (For modern NVIDIA GPUs, RTX)" "CUDA 12.6 (For old NVIDIA GPUs, GTX)" "ROCm (Radeon)" "Intel Arc (beta, may not work)" "CPU"; do
        case $PT_VERSION in
            "CUDA 13.0 (For modern NVIDIA GPUs, RTX)")
                echo "Installing PyTorch with CUDA 13.0..."
                pip3 install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
                break
                ;;
            "CUDA 12.6 (For old NVIDIA GPUs, GTX)")
                echo "Installing PyTorch with CUDA 12.6..."
                pip3 install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
                break
                ;;
            "ROCm (Radeon)")
                echo "Installing PyTorch with ROCm 7.1..."
                pip3 install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
                break
                ;;
            "XPU (Intel Arc / Data Center GPU)")
                echo "Installing PyTorch with XPU support..."
                pip3 install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
                break
                ;;
            CPU)
                echo "Installing CPU-only version of PyTorch..."
                pip3 install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                break
                ;;
            *)
                echo "Invalid option. Please choose 1, 2, 3, 4, or 5."
                ;;
        esac
    done
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Setting the application to executable
chmod +x run_sammie.command

# Install other dependencies
echo "Installing requirements..."
pip3 install -r requirements.txt

# detectron2 (ViTMatte / MEMatte) builds against the installed torch, so it needs
# --no-build-isolation and a C++ compiler.
# On macOS this needs the Xcode command line tools (xcode-select --install).
echo "Installing detectron2..."
if ! pip3 install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833"; then
    echo "WARNING: detectron2 failed to build. Sammie still works; only ViTMatte and MEMatte are unavailable."
fi

# LTX-2 code for the LTX Alpha matting model. It is licensed separately from Sammie
# (LTX-2 Community License), so it is not part of this repository.
echo
read -r -p "Install Lightricks' LTX-2 code for LTX Alpha matting (LTX-2 Community License)? [y/N] " LTX_ANSWER
if [[ "$LTX_ANSWER" =~ ^[Yy]$ ]]; then
    LTX_REF="be6d6d0a69d11a0a399c462ed92606692e5f8697"
    # --no-deps: their dependencies are in requirements.txt, and letting pip resolve
    # torchaudio from PyPI could replace the CUDA build of torch.
    pip3 install --no-deps \
        "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-core" \
        "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-pipelines"
    # Apple Silicon: LTX-2's attention kernel on MPS (an ltx-core dependency skipped by --no-deps).
    if [[ "$OS_TYPE" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        pip3 install "mps-sdpa>=0.2.0"
    fi
fi

echo "Setup complete. Models are not downloaded automatically - see README.md."