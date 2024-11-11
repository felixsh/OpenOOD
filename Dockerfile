# Start from the official NVIDIA CUDA base image for Ubuntu 20.04
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt update && apt dist-upgrade -y && apt install -y \
    sudo \
    curl \
    ca-certificates \
    gnupg2 \
    build-essential \
    bash-completion \
    git \
    nano \
    wget \
    software-properties-common \
    libgl1-mesa-glx

# Create the 'developer' user with UID 1000 and add to 'sudo' group with no password required for sudo commands
RUN useradd --create-home --shell /bin/bash developer && \
    usermod -aG sudo developer && \
    echo 'developer ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer

# Set the working directory to the developer's workspace
RUN mkdir -p /home/developer/workspace 
COPY . /home/developer/workspace
WORKDIR /home/developer/workspace

RUN apt install -y \
    python3 \
    python3-pip \
    cython3 \
    python3-ipython

RUN python3 -m pip install --no-cache-dir --upgrade pipenv

# Setup CUDA
# Add the NVIDIA package repositories
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
#    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
#    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
#    apt update && \
#    apt install -y cuda-12-1 && \
#    apt install -y cuda-toolkit-12-1

# RUN apt-get update \
#     # Optionally, install cuDNN and other CUDA-accelerated libraries here
#     && ln -s cuda-12.1 /usr/local/cuda \
#     && rm -rf /var/lib/apt/lists/*
# CUDA 12.1 is installed in the base image, so we just need to set PATH and LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}


# Configure environment variables
# RUN export PATH=/usr/local/cuda-12.1/bin:${PATH} && \
#     export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
# 
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# RUN pipenv sync

# Define Miniconda version specifics
ARG MINICONDA_VERSION=py39_23.11.0-2
ARG MINICONDA_SHA256=b911ff745c55db982078ac51ed4d848da0170f16ba642822a3bc7dd3fc8c61ba

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /home/developer/miniconda.sh
RUN echo "${MINICONDA_SHA256} */home/developer/miniconda.sh" | sha256sum -c - && \
    /bin/bash /home/developer/miniconda.sh -b -p /opt/conda && \
    rm /home/developer/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Set PATH to include conda
ENV PATH /opt/conda/bin:$PATH

# Install Environment
RUN conda env create -f environment.yml

# Activate
RUN echo "conda activate nc" >> ~/.bashrc
RUN bash -c "source activate nc"

# Install additional packages
# CondaError: Run 'conda init' before 'conda activate'
# RUN conda activate hptr && \
#     conda install transformers && \
#     conda install local-attention

RUN mkdir -p /home/developer/workspace/data
