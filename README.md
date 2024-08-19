<h1>ZipGait</h1>

## Getting started

### 1. Prepare the environments 
  - pytorch >= 1.10
  - torchvision
  - pyyaml
  - tensorboard
  - opencv-python
  - tqdm
  - py7zr
  - kornia
  - einops
  - six

  Install dependenices by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
  ```
  conda install tqdm pyyaml tensorboard opencv kornia einops six -c conda-forge
  conda install pytorch==1.10 torchvision -c pytorch
  ```    
  Or, Install dependencies by pip:
  ```
  pip install tqdm pyyaml tensorboard opencv-python kornia einops six
  pip install torch==1.10 torchvision==0.11
  ```

### 2. Data Preparation
Gait3D, GREW, OU-MVLP, CASIA-B.

### DiffGait Traning
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env --nproc_per_node=1 opengait/main.py --cfgs ./configs/ZipGait/ZipGait_diffgait_Gait3D.yaml --phase train
```
### ZipGait Training
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env --nproc_per_node=1 opengait/main.py --cfgs ./configs/ZipGait/ZipGait_Gait3D.yaml --phase train
```
