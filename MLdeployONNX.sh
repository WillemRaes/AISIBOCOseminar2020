# Deploying ML-models on SBC for runtime:

# Prerequisites Demo on SBC:

# Raspberry Pi 4 flashed with official image from https://www.raspberrypi.org/downloads/raspberry-pi-os/ (lite version)

# SSH access to RPI 

# Necessary packages Linux:

sudo apt-get update
sudo apt-get full-upgrade

sudo apt-get install -y cmake
sudo apt-get install -y build-essential
sudo apt-get install -y python3
sudo apt-get install -y python3-pip
sudo apt-get install -y llvm
sudo apt-get install -y libblas-dev
sudo apt-get install -y python3-scipy
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    python3-dev \
    git \
    tar

sudo su
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade wheel
# Installation could require swap memory

# sudo fallocate -l 1G /swapfile
# sudo chmod 600 /swapfile
# sudo mkswap /swapfile
# sudo swapon /swapfile
# sudo nano /etc/fstab
# add following line to file:
# /swapfile swap swap defaults 0 0

# We are using ONNX runtime for deploying a scikit-learn model
# RPI is ARMv7l architecture, ONNX has no pre built package available for ARM 
# Two possibilities: docker image or build from source
# We build from source native (cross-compile is also possible) 

git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
# Add the following lines to /cmake/CMakeLists.txt due to bug in GCC v8
# string(APPEND CMAKE_CXX_FLAGS " -latomic")
# string(APPEND CMAKE_C_FLAGS " -latomic")

python3 -m pip install flake8
python3 -m pip install numpy

# Build for python with flags
sudo ./build.sh --arm --build_wheel --enable_pybind --config Release --skip_test --update --build_shared_lib

# Takes a while: get coffee..
 
# If the build completed successfully, test on RPI shell with:
# python3
# import onnxruntime 

# Docker steps (for interested reader)
# sudo apt-get install docker
# docker pull onnx/onnx-ecosystem
# docker build . -t onnx/onnx-ecosystem
# docker run -p 8888:8888 onnx/onnx-ecosystem



   
