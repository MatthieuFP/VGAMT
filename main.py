import sys
import os
import json
import random
import argparse
import wandb

import numpy as np

import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.model.model import build_model, reload_model
from src.utils import bool_flag, initialize_exp, custom_collate, gather_tensor

from src.data import CommuteEvaluationDataset, ParallelEvaluationDataset, ParallelImageEvaluationDataset, split_dataset, \
    CommuteEvaluationTextOnlyDataset
from src.trainer import Trainer

# Slurm
import warnings
warnings.filterwarnings("ignore")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./models/",
                        help="Experiment dump path")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Cache directory for huggingface loading")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--other_seed", type=int, default=-1,
                        help="Random seed for weight init/masking/misc (-1 for non-determinism).")
    parser.add_argument("--iter_seed", type=int, default=12345,
                        help="Random seed for data iteration/shuffling (-1 for non-determinism).")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--guided_self_attention", action="store_true",
                        help="Using guided self attention mechanism instead of standard full self-attention")
    parser.add_argument("--mix_xp", type=float, default=0,
                        help="Mix VMLM and MMT experiments - 0 means no VMLM xp, 0.3 means training the model on 30% VMLM")
    parser.add_argument("--multimodal_model", action="store_true",
                        help="Multimodal experiment - if false text-only MT")
    parser.add_argument("--encoder_attn_mask_text_only", action="store_true",
                        help="Decoder embeds can only attend to position related to text embeds in the cross attn.")
    parser.add_argument("--adapters", action="store_true",
                        help="Add adapters in the model")
    parser.add_argument("--freeze_text_parameters", action="store_true",
                        help="Freeze text parameters - learn only (visual params +) adapters")
    parser.add_argument("--prob_mask_text", type=float, default=0.0,
                        help="Proportion of src text to mask")

    # data
    parser.add_argument("--data_path", type=str, default="./data/en-de",
                        help="Data path")
    parser.add_argument("--data_mix_path", type=str, default="",
                        help="Mix data path if mix xp > 0")
    parser.add_argument("--features_path", type=str, default="./data/multi30k/features")
    parser.add_argument("--features_mix_path", type=str, default="",
                        help="Mix features path if mix xp > 0")
    parser.add_argument("--features_type", type=str, default="mdetr",
                        help="to use clip and mdetr, set to mdetr+clip")
    parser.add_argument("--src_lang", type=str, default="en",
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="Target language")

    # batch parameters
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--num_workers", type=int, default=4)

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--grad_l2_norm", type=bool, default=False,
                        help="L2 normalize gradients.")
    parser.add_argument("--epoch_size", type=int, default=10e7,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--min_epoch", type=int, default=-1)
    parser.add_argument("--init_epoch", type=int, default=-1)
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Label smoothing")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=3,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--test_data_set", type=str, default="",
                        help="Name of the evaluation dataset")
    parser.add_argument("--commute_generation", action="store_true",
                        help="Generation mode en CoMMuTE")

    # debug
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--global_rank", type=int, default=-1,
                        help="Multi-GPU - Global rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    parser.add_argument("--reload_optim", action="store_true")
    parser.add_argument("--init_dec_from_enc", action='store_true',
                        help="Initialize missing decoder params from encoder layers.")
    parser.add_argument("--start_new_xp_from_ckpt", action="store_true",
                        help="New experiment from existing checkpoint. Create a new folder instead of the old one.")

    return parser


def main(params):

    # Debug
    if params.debug:
        params.exp_name = "debug"

    # Init distributed training
    if params.local_rank != -1:
        params.global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(params.local_rank)
        params.device = torch.device("cuda", params.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        params.world_size = torch.distributed.get_world_size()
    else:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params.world_size = 1

    # initialize the experiment & wandb
    if params.global_rank <= 0:
        logger = initialize_exp(params)
        wandb.init(project="VGAMT", name=params.exp_name)

    if params.other_seed > -1:
        # deterministic
        torch.manual_seed(params.other_seed)
        torch.cuda.manual_seed(params.other_seed)
        np.random.seed(params.other_seed)
        random.seed(params.other_seed)

    if params.iter_seed == -1:
        # non-deterministic
        params.iter_seed = None

    # Load data
    if not params.eval_only:
        train_set, valid_set, test_set = split_dataset(params)
        train_mix_set = None
        if params.mix_xp:
            train_mix_set = train_set[1]
            train_set = train_set[0]
    else:
        if params.multimodal_model and not params.test_data_set.startswith("commute"):
            test_set = ParallelImageEvaluationDataset(params, split="test")
        elif params.multimodal_model and params.test_data_set.startswith("commute"):
            test_set = CommuteEvaluationDataset(params)
        elif not params.multimodal_model and params.test_data_set.startswith("commute"):
            test_set = CommuteEvaluationTextOnlyDataset(params)
        else:
            test_set = ParallelEvaluationDataset(params, split="test")

    train_sampler, train_mix_sampler, valid_sampler, test_sampler = None, None, None, None
    if params.world_size > 1:
        if not params.eval_only:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set,
                                                                            num_replicas=params.world_size,
                                                                            rank=params.global_rank)

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set,
                                                                       num_replicas=params.world_size,
                                                                       rank=params.global_rank)

    collate_fn = custom_collate if params.multimodal_model else None
    train_loader, train_mix_loader, valid_loader, test_loader = None, None, None, None
    if not params.eval_only:
        train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=params.num_workers, pin_memory=False,
                                  collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=params.batch_size, num_workers=params.num_workers + 4,
                                  shuffle=False, pin_memory=False, sampler=valid_sampler, collate_fn=collate_fn)
        if params.mix_xp:
            train_mix_loader = DataLoader(train_mix_set, batch_size=params.batch_size, num_workers=params.num_workers,
                                      pin_memory=False,
                                      collate_fn=collate_fn)

    test_loader = DataLoader(test_set, batch_size=params.batch_size, num_workers=params.num_workers + 4,
                             shuffle=False, pin_memory=False, sampler=test_sampler, collate_fn=collate_fn)

    data = {"train": train_loader, "train_mix": train_mix_loader, "valid": valid_loader, "test": test_loader}

    # Build model
    model, params.tokenizer = build_model(params)
    params.model = model

    if params.reload_model or params.reload_checkpoint:
        reload_model(params)

    params.model = model.to(params.device)

    if params.local_rank != -1:
        params.model = DDP(params.model, device_ids=[params.local_rank], output_device=params.local_rank,
                           find_unused_parameters=True)

    params.lang2id = params.tokenizer.lang_code_to_id

    if params.global_rank <= 0:
        # Save config
        wandb.config.update(params.config.to_dict())
        wandb.config.update({"batch_size": params.batch_size,
                             "dropout": params.dropout,
                             "optimizer": params.optimizer,
                             "src_lang": params.src_lang,
                             "tgt_lang": params.tgt_lang
                             }, allow_val_change=True)

    # Build Trainer
    trainer = Trainer(data, params)

    # evaluation
    if params.eval_only:
        if params.test_data_set.startswith("commute"):
            if params.multimodal_model:
                if not params.commute_generation:
                    scores = trainer.evaluate_commute()
                else:
                    scores = trainer.evaluate_commute_generation()
            else:
                scores = trainer.evaluate_commute_mt_generation()
        else:
            if params.multimodal_model:
                scores = trainer.evaluate_mmt(mode="test")
            else:
                scores = trainer.evaluate_mt(mode="test")
            
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        sys.exit()

    _iter = 0
    for _ in range(params.max_epoch):

        if params.global_rank <= 0:
            logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        while trainer.n_sentences < trainer.epoch_size:

            if params.mix_xp:
                # Handle asynchronous random generators
                prob_xp = torch.rand(1, device=params.device)
                prob_xp = gather_tensor(prob_xp)[0].item()

            if params.multimodal_model:
                if params.mix_xp:
                    if prob_xp <= params.mix_xp:
                        trainer.vmlm_step(mode="train_mix")
                    else:
                        trainer.mmt_step(mode="train")
                else:
                    trainer.mmt_step(mode="train")

            else:
                if params.mix_xp:
                    if prob_xp <= params.mix_xp:
                        trainer.mlm_step(mode="train_mix")
                    else:
                        trainer.mt_step(mode="train")
                else:
                    trainer.mt_step(mode="train")

            trainer.iter()
            _iter += 1
            torch.cuda.empty_cache()

        if params.global_rank <= 0:
            logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        if params.multimodal_model:
            scores = trainer.evaluate_mmt(mode="valid")
            scores_test = trainer.evaluate_mmt(mode="test")
        else:
            scores = trainer.evaluate_mt(mode="valid")
            scores_test = trainer.evaluate_mt(mode="test")
        
        scores.update(scores_test)

        if params.global_rank <= 0:
            # print / JSON log
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        torch.cuda.synchronize()


if __name__ == "__main__":

    parser = get_parser()
    params = parser.parse_args()

    main(params)