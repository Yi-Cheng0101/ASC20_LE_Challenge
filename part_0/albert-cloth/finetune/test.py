from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import random
import data_util
from data_util import ClothSample
import numpy as np
import torch
import time
import transformers
from transformers import optimization
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from pretrained_albert.modeling import AlbertForCloth
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools
from timeit import default_timer as timer

import pdb

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
            
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='cloth',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='EXP/',
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--cache_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        default=False,
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
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--model_load_dir',
                        type=str,
                        required=True,
                        help="The model.bin directory location")
    args = parser.parse_args()
    
        
    suffix = time.strftime('%Y%m%d-%H%M%S')
    args.output_dir = os.path.join(args.output_dir, suffix)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging = get_logger(os.path.join(args.output_dir, 'log.txt'))
    
    data_file = {'ELE-dev':'ELE-dev'}
    for key in data_file.keys():
        data_file[key] = data_file[key] + '-' + args.bert_model + '.pt'
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logging("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logging("device {} n_gpu {} distributed training {}".format(device, n_gpu, bool(args.local_rank != -1)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    # Prepare model
    model = AlbertForCloth.from_pretrained(args.model_load_dir,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()


    global_step = 0
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logging("***** Running evaluation *****")
        logging("  Batch size = {}".format(args.eval_batch_size))
        valid_data = data_util.Loader(args.data_dir, data_file['ELE-dev'], args.cache_size, args.eval_batch_size, device)
        # Run prediction for full data

        model.eval()
        eval_loss, eval_accuracy, eval_h_acc, eval_m_acc = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, nb_eval_h_examples = 0, 0, 0
        for inp, tgt in valid_data.data_iter(shuffle=False):

            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy, tmp_h_acc, tmp_m_acc = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean() # mean() to average on multi-gpu.
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
                tmp_h_acc = tmp_h_acc.sum()
                tmp_m_acc = tmp_m_acc.sum()

            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            eval_h_acc += tmp_h_acc.item()
            eval_m_acc += tmp_m_acc.item()
            nb_eval_examples += inp[-2].sum().item()
            nb_eval_h_examples += (inp[-2].sum(-1) * inp[-1]).sum().item()
            nb_eval_steps += 1         

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        if nb_eval_h_examples != 0:
            eval_h_acc = eval_h_acc / nb_eval_h_examples
        else:
            eval_h_acc = 0
        eval_m_acc = eval_m_acc / (nb_eval_examples - nb_eval_h_examples)
        result = {'dev_eval_loss': eval_loss,
                  'dev_eval_accuracy': eval_accuracy,
                  'dev_h_acc':eval_h_acc,
                  'dev_m_acc':eval_m_acc,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logging("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logging("  {} = {}".format(key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
