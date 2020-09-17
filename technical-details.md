# Technical details

## TL;DR

In this project I used a sequence-to-sequence model from the transformers library to train a summarization model to generate headlines from the data. Documentation for the same can be found in my [personal fork](https://github.com/SatyaSiddharthDash/transformers) of the transformers library.

## details

The training script provides exact details of the specific trainig and model parameters used. The small version of T5 model was finetuned on the dataset to generate headlines.
The training command I used was:

```shell
python finetune.py \
    --gpus 1 \
    --check_val_every_n_epoch 2 \
    --accumulate_grad_batches 1 \
    --precision 16 \
    --model_name_or_path 't5-small' \
    --cache_dir '$HOME/headlinegen/models' \
    --num_workers 32 \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --output_dir $HOME/headlinegen/data/nytimes_results/ \
    --do_train \
    --do_predict \
    --seed 1024 \
    --max_source_length 1024 \
    --max_target_length 18 \
    --val_max_target_length 18 \
    --test_max_target_length 18 \
    --data_dir ~/headlinegen/data/nytimes/ \
    --task summarization \
    --early_stopping_patience 4

```

The finetuning script includes code for validation and testing as well.
Once trained, the model was loaded as a pretrained model for deployment.