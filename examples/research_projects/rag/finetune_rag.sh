export PYTHONPATH="../":"${PYTHONPATH}"
export HF_HOME="/dccstor/dialog/sfeng/hf_home"
export TOKENIZERS_PARALLELISM=false
YOUR_PROJ_DIR="/dccstor/dialog/sfeng/transformers_doc2dial"
export TRANSFORMERS_CACHE=$YOUR_PROJ_DIR/cache
task=grounding
seg=token
format=two
dpr=dpr_new
MODEL_NAME_OR_PATH=/dccstor/dialog/sfeng/transformers_doc2dial/checkpoints/rag-$dpr
core=1
epoch=15
sourcelen=128
targetlen=50
topn=5
KB_FOLDER=/dccstor/dialog/sfeng/projects/transformers_dialdoc/data_v2/dd_knowledge_dataset-$seg-$dpr
DATA_DIR=/dccstor/dialog/sfeng/projects/transformers_dialdoc/data_v2/dd_$task\_$seg\_$format
config=dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr
python finetune_rag.py \
    --segmentation $seg \
    --data_dir $DATA_DIR \
    --scoring_func linear \
    --cache_dir $YOUR_PROJ_DIR/cache \
    --output_dir $YOUR_PROJ_DIR/checkpoints/$config \
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
    --n_train 10 \
    --n_val 2 \
    --n_test 2 \
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
