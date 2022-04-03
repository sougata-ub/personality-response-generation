#!/bin/sh

#for fname in "$1"/*.pkl
#do
#  echo "Running test for $fname"
#  python3 ./run_additional_test_v2.py --fname "$fname"
#done

#ONLY FOR TRAINING
#for i in $(seq "$1" "$2")
#do
#  echo "Running experiment_num: $i"
#  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port 9999 ./run_training_v2.py --batch_size 16 --num_epochs 15 --learning_rate 0.00002 --experiment_num "$i" --path "/home/sougatas/wikibot_naacl_2022/data/" --num_workers 4 --accum_iter 2 --results_folder "/home/sougatas/wikibot_naacl_2022/results/results_wow_blender_v4/" --model_folder "/home/sougatas/wikibot_naacl_2022/models/models_v2/"
#done

#ONLY FOR GENERATION
#for i in $(seq "$1" "$2")
#do
#  echo "Running experiment_num: $i"
#  python -m torch.distributed.run --nproc_per_node=1 --master_port "$4" ./run_training_v2.py --batch_size 32 --num_epochs 1 --learning_rate 0.00002 --experiment_num "$i" --path "/home/sougatas/wikibot_naacl_2022/data/" --num_workers 4 --accum_iter 1 --results_folder "/home/sougatas/wikibot_naacl_2022/results/all_generation_results_rerun/" --model_folder "/home/sougatas/wikibot_naacl_2022/models/" --device_num "$3" --skip_training "true"
#done

#for fname in "$1"/*.pkl
#do
#  echo "Running test for $fname"
#  python3 ./personality_inference.py --input_pickle_file "$fname" --device_num "$2"
#done

for fname in "$1"/*.pkl
do
  echo "Running test for $fname"
  python3 ./intent_inference.py --input_pickle_file "$fname" --device_num "$2"
done

#INITIAL TRAINING
#echo "Running experiment_num: $1"
#python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port 9999 ./run_training_v2.py --batch_size 32 --num_epochs 8 --learning_rate 0.00002 --experiment_num "$1" --path "/home/sougatas/wikibot_naacl_2022/data/" --num_workers 4 --accum_iter 1 --results_folder /home/sougatas/wikibot_naacl_2022/results/ --model_folder /home/sougatas/wikibot_naacl_2022/models/
#python3 ./run_additional_test.py --experiment_num "$1"