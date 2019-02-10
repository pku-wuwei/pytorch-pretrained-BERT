#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
for task in 'CTB5POS' 'CTB6CWS'  'CTB6POS'  'CTB9POS'  'MSRANER'  'MSRCWS'  'NLPCCCWS'  'OntoNote4NER'  'PKUCWS'  'ResumeNER'  'UD1POS'
do
python ../examples/run_sequence_labeling.py \
  --task_name ${task} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /data/nfsdata/nlp/datasets/sequence_labeling/CN_NER \
  --bert_model /data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/ \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /data/nfsdata/nlp/projects/tmp
done