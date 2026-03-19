

for fold in {0..3}    
do
    echo "start processing"
    python main.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --fold ${fold} --data_type ts --feat_name feature_conch --projector ""
done
python summalize_res.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --data_type ts --feat_name feature_conch  --projector ""

for fold in {0..3}    
do
    echo "start processing"
    python main.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "mask" --fold ${fold} --data_type ts --feat_name feature_conch --projector ""
done
python summalize_res.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "mask" --data_type ts --feat_name feature_conch  --projector ""

for fold in {0..3}    
do
    echo "start processing" 
    python main.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --fold ${fold} --data_type ts --feat_name feature_conch --projector adapter
done
python summalize_res.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --data_type ts --feat_name feature_conch  --projector adapter

for fold in {0..3}    
do
    echo "start processing"
    python main.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --fold ${fold} --data_type ts --feat_name feature_conch --projector "" --sampling_st random
done
python summalize_res.py --method ProtoSum --dataset BRCA --trainer DeconvExp --version "" --data_type ts --feat_name feature_conch  --projector "" --sampling_st random