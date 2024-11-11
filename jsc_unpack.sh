#!/bin/bash
#SBATCH --account=genai-ad
#SBATCH --job-name=unzip_data
#SBATCH --partition=booster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=/p/project/genai-ad/fzi/tas/out/logs/slurm-%A_%a.out
#SBATCH --array=1-72

# srun will no longer read in SLURM_CPUS_PER_TASK and will not inherit option
# --cpus-per-task from sbatch! This means you will explicitly have to specify
export SRUN_CPUS_PER_TASK=4

# Optional to disable the external environment, necessary, if python version is different
module purge

# https://jugit.fz-juelich.de/aoberstrass/bda/ml-pipeline-template/-/blob/main/%7B%7Bcookiecutter.project_name%7D%7D/scripts/train_juwels.sbatch
# export CUDA_VISIBLE_DEVICES=0,1,2,3

dir=$1
config=${dir}job_array_config.txt

container=/p/project/genai-ad/fzi/tas/nc.sif

mode=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
perc=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

srun apptainer exec --nv $container bash -c "cd /p/project1/genai-ad/fzi/tas/OpenOOD/data_openood/images_largescale && bash unpack_imagenet.sh \
    dir=$dir \
    mode=$mode \
    perc=$perc"

wait