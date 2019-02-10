#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
export SQUAD_DIR=/data/nfsdata/nlp/datasets/reading_comprehension/squad/

python ../examples/run_squad.py \
  --bert_model /data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12/  \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file ${SQUAD_DIR}/train-v1.1.json \
  --predict_file ${SQUAD_DIR}/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /data/nfsdata/nlp/projects/tmp/squad
