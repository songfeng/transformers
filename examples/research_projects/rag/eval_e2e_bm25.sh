checkpoint="checkpoint10"
seg=token
task=grounding
split=test
score=original
# dpr=dpr_bi_$seg
dpr=dpr_new
format=two
model=rag_token
topn=5
sourcelen=128
targetlen=50
SFD=/dccstor/dialog/sfeng/projects/transformers_dialdoc
DATA_DIR=$SFD/data_v2/dd_$task\_$seg\_$format
KB_FOLDER=$SFD/data_v2/dd_knowledge_dataset-$seg-$dpr
FD=/dccstor/dialog/sfeng/transformers_doc2dial/checkpoints
MODEL_PATH=$FD/dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr-$score-bm25/$checkpoint
config=dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr-$score-bm25/$checkpoint

jbsub -cores 4+1 -mem 256g -queue x86_1h -require v100 \
-out $SFD/logs_eval/eval_e2e_$config.out \
-err $SFD/logs_eval/eval_e2e_$config.err \
python eval_rag.py \
--model_type rag_token \
--passages_path $KB_FOLDER/my_knowledge_dataset \
--index_path $KB_FOLDER/my_knowledge_dataset_hnsw_index.faiss \
--bm25 $SFD/data_v2/dd_$task\_$seg\_two/doc2dial_$seg.csv \
--n_docs $topn \
--model_name_or_path $MODEL_PATH \
--eval_mode e2e \
--evaluation_set $DATA_DIR/$split.source \
--gold_data_path $DATA_DIR/$split.target \
--gold_data_mode ans \
--eval_batch_size 15 \
--recalculate \
--predictions_path $SFD/results/eval-$config-e2e.txt
#--gold_pid_path $DATA_DIR/$split.pids \
#--eval_all_checkpoints \


