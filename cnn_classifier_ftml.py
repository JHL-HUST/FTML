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

import cnn_model

from cnn_utils import *


def find_synonym(xs, dist_mat, threshold=0.5):
    synonyms = dist_mat[:, :, 0][xs]
    synonyms_dist = dist_mat[:, :, 1][xs]
    synonyms = torch.where(synonyms_dist <= threshold, synonyms, torch.zeros_like(synonyms))
    synonyms = torch.where(synonyms >= 0, synonyms, torch.zeros_like(synonyms)) # [None, Sequence_len, Syn_num]
    return synonyms 


def triplet_loss(xs, dist_mat, embeddings, device, vocab_size, nb_negtive=8, margin=6.0):

    mask = xs.ne(0.0)
    xs = torch.masked_select(xs, mask)
    
    anchor = embeddings[xs] # [b*n, d]

    synonyms = find_synonym(xs, dist_mat).long() # [b*n, k]
    positive = embeddings[synonyms] # [b*n, k, d]
    pos_mask = torch.ge(synonyms, 1).float() # [b*n, k] filter where synonym is [UNK]

    non_synonyms = torch.randint(low=1, high=vocab_size, size=(nb_negtive,), dtype=torch.long).to(device).detach()
    non_synonyms = non_synonyms.unsqueeze(0) # [1, k_n]
    negtive = embeddings[non_synonyms] # [1, k_n, d]
    neg_mask = torch.ne(non_synonyms, xs.unsqueeze(-1)).float() # [b*n, k_n] filter where negtive sample is the anchor

    syn_dis = torch.linalg.norm(anchor.unsqueeze(1) - positive, dim=-1, ord=2) * pos_mask
    nonsyn_dis = torch.linalg.norm(anchor.unsqueeze(1) - negtive, dim=-1, ord=2)

    tpl_loss = torch.mean(syn_dis, dim=-1) - torch.mean(torch.min(torch.zeros_like(nonsyn_dis) + margin, other=nonsyn_dis), dim=-1) + margin # [b*n, ]
    tpl_loss = torch.mean(torch.max(torch.zeros_like(tpl_loss, dtype=torch.float), tpl_loss))

    return tpl_loss, torch.mean(syn_dis), torch.mean(nonsyn_dis)


def load_dataset(args, vocab):

    filename = os.path.join(args.data_dir, "temp", "{}_for_cnn_train.data".format(args.task_name))

    if os.path.exists(filename):
        print("load dataset from exist file.")
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_data=saved['train_data']
        test_data=saved['test_data']
    else:
        train_examples, labels = read_text("%s/train" % args.task_name, args.data_dir)
        seqs, seqs_mask = text_encoder_v2(train_examples, vocab, args.max_seq_length)

        train_data = TensorDataset(torch.tensor(seqs, dtype=torch.long), \
                                    torch.tensor(seqs_mask, dtype=torch.long), \
                                    torch.tensor(labels, dtype=torch.long)) 

        test_examples, test_labels = read_text("%s/test" % args.task_name, args.data_dir)
        test_seqs, test_seqs_mask = text_encoder(test_examples, vocab, args.max_seq_length)
        test_data = TensorDataset(torch.tensor(test_seqs, dtype=torch.long), \
                                    torch.tensor(test_seqs_mask, dtype=torch.long), \
                                    torch.tensor(test_labels, dtype=torch.long))
        
        f = open(filename,'wb')
        saved = {}
        saved['train_data'] = train_data
        saved['test_data'] = test_data
        pickle.dump(saved, f, protocol = 4)
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
    parser.add_argument("--task_name",
                        default="imdb",
                        type=str,
                        # required=True,
                        help="The name of the task to train. `imdb` or `yelp` or `yahoo`")
    parser.add_argument("--model_type",
                        default="CNNModel",
                        type=str,
                        # required=True,
                        help="CNNModel or BiLSTMModel.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")  # 512 for IMDB, 256 for Yelp-5 and Yahoo! Answers
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--freeze_emb",
                        action='store_true',
                        help="Whether to freeze embedding.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=float,
                        help="Total number of training epochs to perform.") # 20 for IMDB, 5 for Yelp-5 and Yahoo! Answers
    parser.add_argument("--max_candidates",
                        default=8,
                        type=int)
    parser.add_argument("--beta",
                        default=1.0,
                        type=float,
                        help="hyperparameter for the training objective")
    parser.add_argument("--nb_negtive",
                        default=8,
                        type=int)
    parser.add_argument("--alpha",
                        default=6.0,
                        type=float,
                        help="hyperparameter for triplet loss.")

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
    parser.add_argument('--vGPU', type=str, default='0', help="Specify which GPUs to use.")
    args = parser.parse_args()
    
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
        device, n_gpu, bool(args.local_rank != -1), False))

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_eval_file = os.path.join(args.output_dir, "train_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(str(vars(args)) + '\n')

    task_name = args.task_name.lower()
    num_labels = num_labels_task[task_name]

    if args.do_train:

        # Prepare data
        vocab, _ = load_dictionary(task_name, args.vocab_size, data_dir=args.data_dir)
        dist_mat = load_dist_mat(task_name, args.vocab_size, data_dir=args.data_dir)
        for stop_word in stop_words:
            if stop_word in vocab:
                dist_mat[vocab[stop_word], :, :] = 0
        dist_mat = torch.tensor(dist_mat[:, :args.max_candidates, :]).to(device)

        train_data, test_data = load_dataset(args, vocab)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        num_train_optimization_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model
        pretrained_emb = load_pretrained_embedding(task_name, args.vocab_size, args.data_dir, 'glove')
        model = getattr(cnn_model, args.model_type)(num_labels, vocab, args.max_seq_length, device, pretrained_emb=pretrained_emb.T, freeze_emb=args.freeze_emb)
        # model.init_weight()
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
        if args.freeze_emb == True:
            model.embs.weight.requires_grad = False
        else:
            model.embs.weight.requires_grad = True
        params = [param for param in model.parameters() if param.requires_grad]


        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.num_train_epochs * 0.75), gamma=0.1)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
    
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        
        loss_fn = torch.nn.CrossEntropyLoss()

        for ind in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, input_labels = batch

                embedded_x = model.input_to_embs(input_ids)
                embedded_x = embedded_x.detach()
                logits = model.embs_to_logit(embedded_x)
                clean_loss = loss_fn(logits, input_labels)

                embeddings = model.get_embeddings()

                sa, syn_dis, non_syndis = triplet_loss(input_ids, dist_mat, embeddings, device, len(vocab), args.nb_negtive, args.alpha)

                loss = clean_loss + args.beta * sa

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # eval during training
                input_labels = input_labels.to('cpu').numpy()
                logits = logits.detach().cpu().numpy()
                tmp_eval_accuracy = accuracy(logits, input_labels)
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            scheduler.step()

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
                for key in sorted(result.keys()):
                    logger.info("%s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write('\n')
                    
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "epoch"+str(ind))
            torch.save(model_to_save.state_dict(), output_model_file)

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                predictions = []

                loss_fn = torch.nn.CrossEntropyLoss()

                for input_ids, input_mask, labels in tqdm(test_dataloader, desc="Evaluating"):
                        
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    labels = labels.to(device)            

                    with torch.no_grad():
                        logits = model.input_to_logit(input_ids)
                        tmp_eval_loss = loss_fn(logits, labels)

                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, labels)

                    predictions.extend(list(np.argmax(logits, axis=1)))

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                result = {'epoch': ind,
                        'eval_loss': eval_loss,
                        'eval_accuracy': eval_accuracy}

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()