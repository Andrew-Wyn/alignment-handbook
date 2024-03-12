#!/bin/bash
#SBATCH --job-name=stf_tuning_en           # Job name
#SBATCH -o /leonardo_work/IscrB_medit/stf_tuning_en-job.out           # Name of stdout output file
#SBATCH -e /leonardo_work/IscrB_medit/stf_tuning_en-job.err           # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=4               # number of threads per task
#SBATCH --time 4:00:00                  # format: HH:MM:SS
#SBATCH --mem 64GB
#SBATCH --gres=gpu:4

#SBATCH -A IscrB_medit

module load profile/deeplrn
module load cuda/12

export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline

source /leonardo/home/userexternal/lmoroni0/__Work/alignment-handbook/.env/bin/activate
ACCELERATE_LOG_LEVEL=info /leonardo/home/userexternal/lmoroni0/__Work/alignment-handbook/.env/bin/accelerate launch \
    --num_processes=4 \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml