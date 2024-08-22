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

## Alternatively

- run the shell script in this directory: `./cuda-installer.sh`
