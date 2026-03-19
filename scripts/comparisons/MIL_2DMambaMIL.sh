#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -P gcf51339

cd /home/acg16920ls/digital_slide
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/12.6/12.6.1
module load cudnn/9.5/9.5.1
source ~/mamba/bin/activate

for dataset in BRCA KIRC LUAD
do
    for fold in {0..3}
    do
        python main.py --method MambaMIL_2D --dataset ${dataset} --trainer "Mamba2DTrainer" --resolution "" --version "stop_sampling" --fold ${fold} --data_type ts 
    done
    python summalize_res.py --method MambaMIL_2D --dataset ${dataset} --trainer "Mamba2DTrainer" --resolution "" --version "stop_sampling" --data_type ts 
done