## Installation

The repository contains submodules, thus please check it out with 
```shell
conda create -n vrobo -y python=3.8
conda activate vrobo
pip install --upgrade pip

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit 

pip install open3d pytorch3d plyfile opencv-python einops e3nn rpyc
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

Download our captured demo data from [Google Drive](https://drive.google.com/file/d/1qvmFvhSha5FnKgFldW8WFAEO50HhQ7GZ/view?usp=drive_link)
```shell
pip install gdown
gdown 1qvmFvhSha5FnKgFldW8WFAEO50HhQ7GZ && unzip vr-robo-dataset.zip
```
The data folder should like this:
```shell
vr-robo-dataset
├── pcd
│   ├── obj1
│   │   └── point_cloud.ply
│   ├── obj2
│   │   └── point_cloud.ply
│   │── ...
├── usd
│   ├── obj1.usd
│   ├── obj2.usd
│   │── ...
```


