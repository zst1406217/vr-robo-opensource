<div align="center">

# VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion

[![Paper](https://img.shields.io/badge/arXiv-2502.01536-brightgreen)](https://arxiv.org/abs/2502.01536) [![Project WebPage](https://img.shields.io/badge/Project-webpage-%23fc4d5d)](https://vr-robo.github.io/)

</div>

> [VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion](https://vr-robo.github.io/) \
> Shaoting Zhu*, Linzhan Mou*, Derun Li, Baijun Ye, Runhan Huang, Hang Zhao† \
> RA-L 2025

<div align="center">
    <img src="teaser.png" alt="VR-Robo Teaser" style="max-width: 100%;" />
</div>

## Installation
We use two different environments for the Isaac Lab and the renderer. Please follow the instructions in each directory to finish the installation.
- For Isaac Lab environment, please refer to [vrrobo_isaaclab/README.md](vrrobo_isaaclab/README.md) for installation instructions.
- For the renderer environment, check out [vrrobo_renderer/README.md](vrrobo_renderer/README.md) to complete the setup.

## Usage
To run the demo, you need to first start the render server in the `vrrobo_renderer` directory:
```shell
conda activate vr-robo-renderer
python render_server.py
```
Then, start a new terminal in the `vrrobo_isaaclab` directory, you can run the following command to play the demo:
```shell
conda activate vr-robo-isaaclab
python scripts/rsl_rl/play_gs.py --task go2_gs_play
```
If you want to train the model, you can run:
```shell
conda activate vr-robo-isaaclab
python scripts/rsl_rl/train_gs.py --task go2_gs --headless
```

## 🧷 News

- **[2025-06-14]** We release the training and playing code in simulation.

- **[2025-05-11]** Our paper is accepted by RA-L 2025. We will release code in this repo.

- **[2025-02-01]** Paper released on arXiv.


## 📝 Citation

You can find our paper on [arXiv](https://arxiv.org/pdf/2502.01536).

If you find this code or find the paper useful for your research, please consider citing:

```
@article{zhu2025vr,
  title={VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion},
  author={Zhu, Shaoting and Mou, Linzhan and Li, Derun and Ye, Baijun and Huang, Runhan and Zhao, Hang},
  journal={arXiv preprint arXiv:2502.01536},
  year={2025}
}
```
