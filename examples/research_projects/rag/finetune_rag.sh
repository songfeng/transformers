export PYTHONPATH="../":"${PYTHONPATH}"
export HF_HOME="/dccstor/dialog/sfeng/hf_home"
export TOKENIZERS_PARALLELISM=false
YOUR_PROJ_DIR="/dccstor/dialog/sfeng/transformers_doc2dial"
export TRANSFORMERS_CACHE=$YOUR_PROJ_DIR/cache
task=grounding
seg=structure
score=linear3
format=two
dpr=dpr_new
# dpr=dpr_bi_$seg
MODEL_NAME_OR_PATH=/dccstor/dialog/sfeng/transformers_doc2dial/checkpoints/rag-$dpr
# dpr=""
# MODEL_NAME_OR_PATH='facebook/rag-token-base'
core=1
epoch=10
sourcelen=128
targetlen=50
topn=5
KB_FOLDER=/dccstor/dialog/sfeng/projects/transformers_dialdoc/data_v2/dd_knowledge_dataset-$seg-$dpr
DATA_DIR=/dccstor/dialog/sfeng/projects/transformers_dialdoc/data_v2/dd_$task\_$seg\_$format
config=dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr-$score
jbsub -cores 4+$core -mem 128g -queue x86_24h -require v100 \
-out logs/$config.out \
-err logs/$config.err \
python finetune_rag.py \
    --segmentation $seg \
    --data_dir $DATA_DIR \
    --scoring_func $score \
    --cache_dir $YOUR_PROJ_DIR/cache \
    --output_dir output/$config \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_token \
    --index_name custom \
    --passages_path $KB_FOLDER/my_knowledge_dataset \
    --index_path $KB_FOLDER/my_knowledge_dataset_hnsw_index.faiss \
    --fp16 \
    --profile \
    --do_train \
    --do_predict \
    --gpus $core \
    --n_train 12 \
    --n_val 3 \
    --n_test 3 \
    --n_docs $topn \
    --train_batch_size 6 \
    --eval_batch_size 1 \
    --max_combined_length 300 \
    --max_source_length $sourcelen \
    --max_target_length $targetlen \
    --val_max_target_length $targetlen \
    --test_max_target_length $targetlen \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs $epoch \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 
