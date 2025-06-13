## Installation

First install Isaac Sim v4.2 and Isaac Lab v1.3.0 following the [NVIDIA Isaac Sim documentation](https://isaac-sim.github.io/IsaacLab/v1.3.0/source/setup/installation/pip_installation.html). Please do not clone the Isaac Lab repository in `./vrrobo_isaaclab` directory. You can clone and install it in any directory you like. We recommend you to install it in `./VR-Robo` directory.

```shell
conda create -n vr-robo-isaaclab python=3.10
conda activate vr-robo-isaaclab
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
git clone git@github.com:isaac-sim/IsaacLab.git -b v1.3.0
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

Then install this repository the required packages. In `./vrrobo_isaaclab` directory, run the following commands:
```shell
python -m pip install -e exts/vrrobo_isaaclab
pip install -e exts/rsl_rl
pip install gymnasium==0.29.0 rpyc
```

## Training
```shell
python scripts/rsl_rl/train_gs.py --task go2_gs --headless
```

## Testing
```shell
python scripts/rsl_rl/play_gs.py --task go2_gs_play --headless
```

## Trouble Shooting
If you can't stop the process, run: 
```shell
killall -9 pt_main_thread
```
