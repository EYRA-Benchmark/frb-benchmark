#!/bin/bash
# This file was used to prepare a AWS p3.2xlarge machine running Ubuntu 18.04 for running Docker containers with GPU support.
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common \
    build-essential

wget http://us.download.nvidia.com/tesla/440.33.01/NVIDIA-Linux-x86_64-440.33.01.run
chmod a+x ./NVIDIA-Linux-x86_64-440.33.01.run
sudo sh ./NVIDIA-Linux-x86_64-440.33.01.run -q -a -n -s
sudo modprobe nvidia
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -a -G docker ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "Done, user 'ubuntu' has been added to the Docker group. You'll need to log out & in for this to take effect."
echo "To test GPU support is working in a container, run: 'docker run --rm --gpus all nvidia/cuda:9.0-base nvidia-smi'"

