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


data_type=ts
 
for dataset in BRCA
do
    for fold in {0..3}
    do
        python main.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
            --version "scratch" --fold ${fold} --data_type ${data_type} >& reg.log
    done
    python summalize_res.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
        --version "scratch" --data_type ${data_type}
done

for dataset in BRCA
do
    for fold in {0..3}
    do
        python main.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
            --version "wocorrection" --fold ${fold} --data_type ${data_type} >& reg.log
    done
    python summalize_res.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
        --version "wocorrection" --data_type ${data_type}
done

for dataset in BRCA
do
    for fold in {0..3}
    do
        python main.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
            --version "wocorrection_static" --fold ${fold} --data_type ${data_type} >& reg.log
    done
    python summalize_res.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
        --version "wocorrection_static" --data_type ${data_type}
done

for dataset in BRCA
do
    for fold in {0..3}
    do
        python main.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
            --version "static" --fold ${fold} --data_type ${data_type} >& reg.log
    done
    python summalize_res.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
        --version "static" --data_type ${data_type}
done

for dataset in BRCA
do
    for fold in {0..3}
    do
        python main.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
            --version "1reg_mse_reg_1e3" --fold ${fold} --data_type ${data_type} >& reg.log
    done
    python summalize_res.py --method ProtoSum --dataset ${dataset} --trainer DeconvExp \
        --version "1reg_mse_reg_1e3" --data_type ${data_type}
done