srun -K --ntasks=1 --gpus-per-task=1 \
--cpus-per-gpu=4 \
-p A100-80GB \
--mem=512G \
--container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
--container-workdir=`pwd` \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
--mail-type=ALL \
--mail-user=ankit.agrawal@dfki.de \
--job-name debug \
--time=300 \
--pty bash


# pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"



# pip install "unsloth[cu124-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.03-py3.sqsh \
# --container-image=/netscratch/agrawal/pytorch1.13cu118.sqsh \

# pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"s

# --time=300 \


# srun -K -p V100-32GB-SDS --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=40GB --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` --container-image=/netscratch/agrawal/pytorch1.13cu118.sqsh --container-workdir=`pwd` --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" --mail-type=ALL --mail-user=ankit.agrawal@dfki.de --job-name test --time=1200 --pty baash
# srun -K -p V100-32GB-SDS --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=40GB --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` --container-image=/netscratch/agrawal/pytorch1.13cu118.sqsh --container-workdir=`pwd` --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" --mail-type=ALL --mail-user=ankit.agrawal@dfki.de --job-name train

#--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.10-py3.sqsh \
#--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.03-py3.sqsh \
#--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.01-py3.sqsh \
#--container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh

# http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/monitoring-your-jobs/

# ssh agrawal@login1.pegasus.kl.dfki.de

# srun -K -p V100-32GB-SDS --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=40GB --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.01-py3.sqsh --container-workdir=`pwd` --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" --mail-type=ALL --mail-user=ankit.agrawal@dfki.de --job-name test --time=4300 python src/lasercut_gnn/train_mgn_lasercut_linear2dsynth_testdata.py
# python3 src/lasercut_gnn/train_mgn_lasercut_linear2dsynth_testdata.py

