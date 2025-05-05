#!/bin/bash
#SBATCH --time=48:00:00 # Run time
#SBATCH --nodes 1  # Number of reaquested nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem 400G
#SBATCH -c 54
#SBATCH -p gpu
#SBATCH --gres=gpu:H100.80gb:1   # GPU type and number
#SBTACH --job-name example_job_name
#SBATCH --error=Example_job_error.o%j
#SBATCH --output=Example_job_output.o%j
#SBATCH --requeue
#SBATCH --mail-user=user_name@uni-osnabrueck.de

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL

#SBATCH --signal=SIGTERM@90
echo "running in shell: " "$SHELL"

export NCCL_SOCKET_IFNAME=lo

export XLA_FLAGS="--xla_gpu_cuda_data_dir=<location of XLA>"
export TMPDIR='<temp location>'

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case
## term handler the function is executed once the job gets the TERM signal

spack load miniconda3

eval "$(conda shell.bash hook)"
conda activate thesis2



srun example_job.py
