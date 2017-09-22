#!/bin/sh
THEANO_FLAGS='floatX=float32,device=gpu0,optimizer=fast_run'   python main.py --train_file ../data/wikitext-103/idx_wiki.train.tokens \
            --valid_file ../data/wikitext-103/idx_wiki.valid.tokens \
            --test_file ../data/wikitext-103/idx_wiki.test.tokens \
            --model_dir './model/parameters_57920.0.pkl' \
            --vocab_size 267736 \
            --batch_size 5 \
            --goto_line 20000
