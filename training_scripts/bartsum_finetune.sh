export PYTHONPATH=$PYTHONPATH:$HOME/headlinegen/transformers_clone/examples
export PYTHONPATH=$PYTHONPATH:$HOME/headlinegen/transformers_clone/examples/seq2seq
export PYTHONPATH=$PYTHONPATH:$HOME/headlinegen
export PYTHONPATH=$PYTHONPATH:$HOME/headlinegen/transformers_clone

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