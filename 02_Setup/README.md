# Setup Instructions

1. first do a `sudo apt update && sudo apt upgrade -y && sudo apt autoremove` then pop over to [downloads](https://developer.nvidia.com/cuda-downloads)
2. fill in the following settings that match the device you'll be doing this course on: Operating System
   - Architecture
   - Distribution
   - Version
   - Installer Type
3. you'll have to run a command very similar to the one below in the "runfile section"

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

4. in the end, you should be able to run `nvcc --version` and get info about the nvidia cuda compiler (version and such).
   also run `nvidia-smi` to ensure nvidia recognizes your cuda version and connected GPU

5. If `nvcc` doesn't work, run `echo $SHELL`. If it says bin/bash, add the following lines to the ~/.bashrc rile. If it says bin/zsh, add to the ~/.zshrc file. 
```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```
do `source ~/.zshrc` or `source ~/.bashrc` after this then try `nvcc -V` again

## Alternatively

- run the shell script in this directory: `./cuda-installer.sh`

## For [WSL2](https://medium.com/@omkarpast/technical-documentation-for-clean-installation-of-ubuntu-cuda-cudnn-and-pytorch-on-wsl2-9b265a4b8821)
