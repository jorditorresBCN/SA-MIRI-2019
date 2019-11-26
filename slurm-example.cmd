#!/bin/bash
#SBATCH --job-name hor
#SBATCH -D /gpfs/home/sam14/sam14010/horovod_2019
#SBATCH --output /gpfs/home/sam14/sam14010/horovod_2019/jobs/%j.out
#SBATCH --error /gpfs/home/sam14/sam14010/horovod_2019/jobs/%j.err
#SBATCH --nodes 1
#SBATCH --tasks-per-node 4
#SBATCH --cpus-per-task 40
#SBATCH --gres gpu:4
#SBATCH --time 00:20:00
#SBATCH --qos=debug


module purge; module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 fftw/3.3.8 ffmpeg/4.2.1 atlas/3.10.3 scalapack/2.0.2 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML

python tf_keras_mnist.py --epochs=2
