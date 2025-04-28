
export WANDB_PROJECT=augmented_unityv2
export PYTHONPATH=${PYTHONPATH}:`pwd`
export TOKENIZERS_PARALLELISM=false

ExpName="deberta_v3_base_msmarco_nc2_5M_bow_cluster4096"
# OUTPUT_DIR="/data_ecstorage/MINDER/experiments/rebuttal/cluster/$ExpName"
OUTPUT_DIR="data/experiments/$ExpName"
DATA="tmp10k.tsv"
# DATA="/data_ecstorage/MINDER/data/training_data/MSMARCO_title_body_query3/nc2_train_deberta_v3_base_msmarco_pq_5M.tsv"

mkdir -p ${OUTPUT_DIR}
deepspeed --num_gpus 16 clover/run_scripts/run_trainer_extreme_nar.py \
    --model_name_or_path /data_ecstorage/MINDER/experiments/MSMARCO/akash_opt_H100_deberta_v3_base_5M_bow_ms_no_downproj \
    --cluster_vectors_path /data_ecstorage/MINDER/experiments/rebuttal/cluster/msmarco_5M_bow_clusters_kmeans4096.npy \
    --model_type deberta_bow_cluster_train \
    --tokenizer_name microsoft/deberta-v3-base \
    --target_tokenizer_name /data_ecstorage/MINDER/mstmp/vocabs/msmarco_pseudo_queries_5M_special.vocab \
    --target_tokenizer_lib tokenmonster \
    --data_schema query,keyword \
    --pretokenized \
    --input_column query \
    --target_column keyword \
    --preprocessing_num_workers 256 \
    --pad_to_max_length \
    --max_source_length 16 \
    --max_target_length 8 \
    --num_clusters 4096 \
    --label_smoothing_factor 0.0 \
    --do_train \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --bf16 \
    --num_train_epochs 5 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${OUTPUT_DIR}/tensorboard \
    --seed 42 \
    --dataloader_num_workers 16 \
    --logging_steps 1 \
    --train_file ${DATA} \
    --per_device_train_batch_size 16 \
    --num_keywords_per_query 40 \
    --gradient_accumulation_steps 1 \
    --deepspeed ./ds_configs/ds_config_z1.json \
    --ddp_timeout 57600 \
    --report_to wandb \
    --run_name $ExpName \
    2>&1 | tee ${OUTPUT_DIR}/log_train.txt
