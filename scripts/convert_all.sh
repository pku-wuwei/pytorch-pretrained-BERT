#!/usr/bin/env bash
export ddir=/data/nfsdata/nlp/BERT_BASE_DIR/
for d in $(ls ${ddir})
do
    python ../pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ${ddir}/${d}/bert_model.ckpt \
  --bert_config_file ${ddir}/${d}/bert_config.json \
  --pytorch_dump_path ${ddir}/${d}/pytorch_model.bin
done
