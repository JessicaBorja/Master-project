echo 'cd /home/' >> ~/.bashrc
apt-get -y update
apt-get -y upgrade
apt-get install -y build-essential
apt-get install wget
apt-get -y install cuda
apt-get install -y git
apt install libx11-dev

# Install cuda toolkit 11.3
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update

# Install conda environment
source ~/.bashrc
conda create -n vapo python=3.8
conda activate vapo

# Install torch
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install hough voting layer
cd /home/
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror/
mkdir build/
cd build/
cmake ..
make install
cd /home/vapo/vapo/affordance/hough_voting/
python setup.py install

# Install pybullet
cd /home/
git clone https://github.com/bulletphysics/bullet3.git
cd bullet3
wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow.cpp
pip install numpy
pip install -e .

# Install vapo
git clone https://github.com/mees/vapo.git
cd /home/vapo/
pip install -e .
cd ./VREnv
pip install -e .
