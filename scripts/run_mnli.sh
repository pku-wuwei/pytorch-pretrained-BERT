#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export GLUE_DIR=/data/nfsdata/nlp/datasets/glue_data/

python ../examples/run_classifier.py \
  --task_name MNLI \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${GLUE_DIR}/MNLI \
  --bert_model /data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12/ \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /data/nfsdata/nlp/projects/tmp/MNLI
