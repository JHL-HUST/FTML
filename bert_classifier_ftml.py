
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences


from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert_model import BertForClassifier, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer, load_vocab
from optimization import BertAdam, warmup_linear
from synonym_selector import EmbeddingSynonym

from bert_utils import *
from cnn_utils import load_dictionary, load_dist_mat, stop_words
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json


def update_vocab_and_embedding(args, tokenizer, attack_vocab, updated_bert_path):

    if not Path(updated_bert_path).is_dir():
        os.mkdir(updated_bert_path)
    elif len(os.listdir(updated_bert_path)) != 0:
        print("updated bert files have been existed.")
        return
    
    print("supplement the bert vocab with actual words!")

    state_dict = torch.load(os.path.join(args.data_dir, args.init_checkpoint_file))
    embeddings = state_dict['bert.embeddings.word_embeddings.weight'].numpy()
    words = list(tokenizer.vocab.keys())
    new_embeddings = list(embeddings)
    for w in tqdm(list(attack_vocab.keys())[1:], total=len(attack_vocab)-1):
        if w.isalpha() and w not in tokenizer.vocab:
            words.append(w)
            new_emb = np.mean(embeddings[np.array([tokenizer.vocab[piece] for piece in tokenizer.tokenize_token(w)])], axis=0)
            new_embeddings.append(new_emb)
    new_embeddings = np.array(new_embeddings)
    print("new embedding shape: ", new_embeddings.shape)
    print("new vocab size: ", len(words))

    # save new vocab
    with open(os.path.join(updated_bert_path, "bert-base-uncased-vocab.txt"), 'a') as f:
        for w in words:
            f.write(w + "\n")
    
    # save new checkpoint
    state_dict['bert.embeddings.word_embeddings.weight'] = torch.tensor(new_embeddings)
    torch.save(state_dict, os.path.join(updated_bert_path, "pytorch_model.bin"))
    
    # save new config
    config = None
    with open(os.path.join(args.data_dir, args.init_config_file), 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['vocab_size'] = len(words)
    with open(os.path.join(updated_bert_path, "bert_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f)


def update_synonym_matrix(tokenizer, synonym_selector, attack_vocab):
    synonym_matrix = [[] for _ in range(len(tokenizer.vocab))] # {bert ids: [bert ids of synonym list]}
    for word, wid in attack_vocab.items():
        if (wid == 0 or word not in tokenizer.vocab):
            continue
        synonyms = synonym_selector.find_synonyms(word)
        synonyms_ids = []
        for w in synonyms:
            if w in tokenizer.vocab:
                synonyms_ids.append(tokenizer.vocab[w])
        synonym_matrix[tokenizer.vocab[word]].extend(synonyms_ids)
    synonym_matrix = pad_sequences(
        synonym_matrix, padding="post", truncating="post", value=0
    )
    print("synonym matrix shape", synonym_matrix.shape)
    return synonym_matrix


def convert_to_bert_id(tokenizer, attack_vocab):
    mapper = []
    for word, idx in attack_vocab.items():
        if (word in tokenizer.vocab):
            mapper.append(tokenizer.vocab[word])
        else:
            mapper.append(tokenizer.vocab['[UNK]'])
    return mapper


def load_dataset(args, tokenizer):
    
    if not Path(os.path.join(args.data_dir, "temp")).is_dir():
        os.mkdir(os.path.join(args.data_dir, "temp")) 

    filename = os.path.join(args.data_dir, "temp/{}_for_tpl_v5_{}.data".format(args.task_name, args.vocab_size))

    if os.path.exists(filename):
        print("load dataset from exist file.")
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_data=saved['train_data']
        test_data=saved['test_data']
    else:
        train_examples = get_examples_for_bert("%s/train" % args.task_name, args.data_dir)
        test_examples = get_examples_for_bert("%s/test" % args.task_name, args.data_dir)
        
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, \
                                   all_segment_ids, all_label_ids) 

        test_features = convert_examples_to_features(
            test_examples, args.max_seq_length, tokenizer)        
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, \
                                   all_segment_ids, all_label_ids) 

        f = open(filename,'wb')
        saved = {}
        saved['train_data'] = train_data
        saved['test_data'] = test_data
        pickle.dump(saved, f)
        f.close()

    return train_data, test_data


def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, # required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--init_checkpoint_file", default="pytorch_model.bin", type=str)
    parser.add_argument("--init_config_file", default="bert_config.json", type=str)
    parser.add_argument("--task_name",
                        default="imdb",
                        type=str,
                        # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="cache",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vGPU", 
                        type=str, 
                        default=None, 
                        help="Specify which GPUs to use.")
    parser.add_argument("--max_candidates",
                        default=8,
                        type=int)
    parser.add_argument("--beta",
                        default=1.0,
                        type=float)
    parser.add_argument("--alpha",
                        default=1.0,
                        type=float)
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--loss_mode",
                        default="triplet",
                        type=str)
    parser.add_argument("--norm",
                        default=2,
                        type=float)

    # The following settings are not important
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.vGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vGPU

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()
    num_labels = num_labels_task[task_name]

    init_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    vocab, inv_vocab = load_dictionary(task_name, args.vocab_size, data_dir=args.data_dir)

    updated_bert_path = os.path.join(args.data_dir, "aux_files/update_bert_{}_{}".format(args.task_name, args.vocab_size))
    update_vocab_and_embedding(args, init_tokenizer, vocab, updated_bert_path)

    tokenizer = BertTokenizer(os.path.join(updated_bert_path, "bert-base-uncased-vocab.txt"), do_lower_case=args.do_lower_case, max_len=512)

    if args.do_train:
        # Prepare synonyms
        dist_mat = load_dist_mat(task_name, args.vocab_size, data_dir=args.data_dir)
        for stop_word in stop_words:
            if stop_word in vocab:
                dist_mat[vocab[stop_word], :, :] = 0
        dist_mat = dist_mat[:, :args.max_candidates, :]
        synonym_selector = EmbeddingSynonym(args.max_candidates, vocab, inv_vocab, dist_mat, threshold=0.5)
        synonyms_matrix = update_synonym_matrix(tokenizer, synonym_selector, vocab)
        synonyms_matrix = torch.nn.Parameter(torch.tensor(synonyms_matrix, dtype=torch.long, requires_grad=False), requires_grad=False)
        id_mapper = convert_to_bert_id(tokenizer, vocab)
        id_mapper = torch.nn.Parameter(torch.tensor(id_mapper, dtype=torch.long, requires_grad=False), requires_grad=False)

        # Prepare data
        train_data, test_data = load_dataset(args, tokenizer)
        logger.info("train data len = %d", len(train_data))
        num_train_optimization_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
        model = BertForClassifier.from_pretrained(updated_bert_path, num_labels=num_labels, synonyms_matrix=synonyms_matrix, id_mapper=id_mapper)
        if args.fp16:
            model.half()
        model.to(device)

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                bias_correction=False,
                                max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
    
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        # tb_writer = SummaryWriter(args.output_dir) 

        for ind in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch 

                logits, loss, clean_loss, sa, pos_distance, neg_distance = model(input_ids, 
                                        attention_mask=input_mask, labels=label_ids,
                                        token_type_ids=segment_ids, optimizer=optimizer, 
                                        beta=args.beta, alpha=args.alpha, mode=args.loss_mode, norm=args.norm, vocab_size=args.vocab_size)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    clean_loss = clean_loss.mean()
                    sa = sa.mean()
                    pos_distance = pos_distance.mean()
                    neg_distance = neg_distance.mean()
                
                # tb_writer.add_scalar("clean_loss", clean_loss, global_step)
                # tb_writer.add_scalar("synonym_loss", sa, global_step)
                # tb_writer.add_scalar("loss", loss, global_step)
                # tb_writer.add_scalar("pos_distance", pos_distance, global_step)
                # tb_writer.add_scalar("neg_distance", neg_distance, global_step)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()


                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # eval during training
                label_ids = label_ids.to('cpu').numpy()
                logits = logits.detach().cpu().numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {
                    'epoch': ind,
                    'train_accuracy': eval_accuracy,
                    'train_loss': loss
                    }

            output_eval_file = os.path.join(args.output_dir, "train_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Training results *****")
                writer.write("epoch"+str(ind)+'\n')
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write('\n')
                    
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "epoch"+str(ind)+"_"+WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                result, _ = evaluate(test_dataloader, model, device, ind)

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))


def evaluate(eval_dataloader, model, device, epoch):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    predictions = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)            

        with torch.no_grad():
            logits, tmp_eval_loss = model(input_ids, attention_mask=input_mask, labels=label_ids, token_type_ids=segment_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        predictions.extend(list(np.argmax(logits, axis=1)))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {'epoch': epoch,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy}

    return result, predictions

if __name__ == "__main__":
    main()