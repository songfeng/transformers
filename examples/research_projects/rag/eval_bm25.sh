checkpoint="checkpoint10"
seg=token
task=grounding
split=val
score=-original
# dpr=dpr_bi_$seg
dpr=dpr_new
format=two
model=rag_token
topn=10
sourcelen=128
targetlen=50
SFD=/dccstor/dialog/sfeng/projects/transformers_dialdoc
DATA_DIR=$SFD/data_v2/dd_$task\_$seg\_$format
KB_FOLDER=$SFD/data_v2/dd_knowledge_dataset-$seg-$dpr
FD=/dccstor/dialog/sfeng/transformers_doc2dial/checkpoints
MODEL_PATH=$FD/dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr$score-bm25/$checkpoint
config=dd-$seg-$task-$sourcelen-$targetlen-$format-$dpr$score-$split-$checkpoint-bm25

jbsub -cores 4+1 -mem 128g -queue x86_1h -require v100 \
-out $SFD/logs_eval/eval_re_$config.out \
-err $SFD/logs_eval/eval_re_$config.err \
python examples/rag/eval_rag.py \
--model_type $model \
--passages_path $KB_FOLDER/my_knowledge_dataset \
--index_path $KB_FOLDER/my_knowledge_dataset_hnsw_index.faiss \
--bm25 $SFD/data_v2/dd_$task\_$seg\_two/doc2dial_$seg.csv \
--n_docs $topn \
--model_name_or_path  $MODEL_PATH \
--eval_mode retrieval \
--k 1 \
--evaluation_set $DATA_DIR/$split.source \
--gold_data_path $DATA_DIR/$split.titles \
--gold_pid_path $DATA_DIR/$split.pids \
--gold_data_mode ans \
--eval_batch_size 40 \
--recalculate \
# --eval_all_checkpoints \
--predictions_path $SFD/results/eval-$config-re.txt