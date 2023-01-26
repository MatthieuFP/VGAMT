import os
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as D
from torch.cuda.amp import autocast, GradScaler

from .optim import get_optimizer
from .utils import to_cuda, eval_moses_bleu, gather_tensor, pad_tensor
from tqdm import tqdm

logger = getLogger()
mbart_langs = {'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'fr': 'fr_XX'}


def gather_tensor(v):
    if v is None:
        return None

    gather_dest = [torch.empty_like(v) * i for i in range(D.get_world_size())]  # list where each element is [N x H_DIM]
    D.all_gather(gather_dest, v.contiguous())  # as far as i recall, this loses gradient information completely

    gather_dest[D.get_rank()] = v  # restore tensor with gradient information
    gather_dest = torch.cat(gather_dest)

    # gather_dest is now a tensor of [(N*N_GPUS) x H_DIM], as if you ran everything on one GPU, except only N samples
    # corresponding to GPU i inputs will have gradient information
    return gather_dest


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """

        self.params = params
        self.data = data
        self.iterators = {}
        self.model = params.model
        self.scaler = GradScaler()

        # epoch / iteration size
        self.epoch_size = params.epoch_size

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16

        # set optimizers
        self.set_optimizers()

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0 if self.params.init_epoch == -1 else self.params.init_epoch
        self.n_iter = 0
        self.n_total_iter = 0 if self.params.init_epoch == -1 else self.params.n_total_iter
        self.n_sentences = 0

        self.stats = OrderedDict(
            [('processed_s', 0)] +
            [('MT-%s-%s' % (params.src_lang, params.tgt_lang), [])] +
            [('VMLM-loss-%s' % (params.src_lang), [])] +
            [('VMLM-acc-%s' % (params.src_lang), [])] +
            [('MLM-loss-%s' % (params.src_lang), [])] +
            [('MLM-acc-%s' % (params.src_lang), [])]
        )

        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for k, p in self.params.model.named_parameters():
            if p.requires_grad:
                named_params.append([k, p])

        self.parameters['model'] = [p for k, p in named_params]

        # log
        for k, v in self.parameters.items():
            if self.params.global_rank <= 0:
                logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # Reload if necessary
        if self.params.reload_optim:
            self.optimizers['model'].load_state_dict(params.optim_from_ckpt)

        # log
        if self.params.global_rank <= 0:
            logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                loss.backward()
                if params.grad_l2_norm:
                    for name in names:
                        for p in filter(lambda x: x.grad is not None, self.parameters[name]):
                            p.grad.data.div_(p.grad.data.norm(2))
                elif params.clip_grad_norm > 0:
                    for name in names:
                        clip_grad_norm_(self.parameters[name], params.clip_grad_norm)

                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss.backward()

        # AMP optimization
        else:
            self.scaler.scale(loss).backward()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    for optimizer in optimizers:
                        self.scaler.unscale_(optimizer)
                    for name in names:
                        clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
                for optimizer in optimizers:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        if self.params.global_rank <= 0:
            self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - ".format(
            self.stats['processed_s'] * 1.0 / diff
        )  # self.stats['processed_w'] * 1.0 / diff
        self.stats['processed_s'] = 0
        # self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.global_rank <= 0:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        logger.warning(f"Saving model parameters ...")
        if self.params.xp_type in ["vmlm", "mlm"] and not self.params.encdec_vmlm:
            try:
                enc_parameters = self.params.encoder.module.encoder.state_dict()
            except:
                enc_parameters = self.params.encoder.encoder.state_dict()
            self.params.model.model.encoder.state_dict().update(enc_parameters)

        data["model"] = self.params.model.state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data['optimizer'] = self.optimizers[name].state_dict()  # 'f{name}_optimizer

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = self.params.reload_model
        if not self.params.reload_model:
            return
        else:
            assert os.path.isfile(checkpoint_path)

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload main metrics
        if self.params.eval_only:
            self.epoch = data["epoch"]
        else:
            self.epoch = data['epoch'] + 1 if not self.params.start_new_xp_from_ckpt else 0
        self.n_total_iter = data['n_total_iter'] if not self.params.start_new_xp_from_ckpt else 0
            
        if not self.params.start_new_xp_from_ckpt:
            self.best_metrics = data['best_metrics']
            self.best_stopping_criterion = data['best_stopping_criterion']
            logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")
        else:
            logger.warning("Checkpoint reloaded. Start new experiment")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.global_rank <= 0:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch, include_optimizers=True)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.global_rank <= 0:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric, include_optimizers=True)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.params.global_rank <= 0:
            if self.stopping_criterion is not None and \
                    (not self.stopping_criterion[0].endswith('_mt_bleu') or not self.stopping_criterion[0].endswith(
                        '_mmt_bleu')):
                metric, biggest = self.stopping_criterion
                assert metric in scores, metric
                factor = 1 if biggest else -1
                if factor * scores[metric] > factor * self.best_stopping_criterion:
                    self.best_stopping_criterion = scores[metric]
                    logger.info("New best validation score: %f" % self.best_stopping_criterion)
                    self.decrease_counts = 0
                else:
                    logger.info("Not a better validation score (%i / %i)."
                                % (self.decrease_counts, self.decrease_counts_max))
                    self.decrease_counts += 1
                if self.decrease_counts > self.decrease_counts_max and self.epoch > self.params.min_epoch:
                    logger.info("Stopping criterion has been below its best value for more "
                                "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                    if self.params.world_size > 1 and 'SLURM_JOB_ID' in os.environ:
                        os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                    sys.exit()

        self.epoch += 1
        if self.epoch < self.params.min_epoch:
            self.decrease_counts = 0

    def get_batch(self, dataset):
        """
        Return a batch of sentences from a dataset.
        """
        iterator = self.iterators.get((dataset), None)
        if iterator is None:
            iterator = iter(self.data[dataset])
            self.iterators[dataset] = iterator
        try:
            x = next(iterator)
        except StopIteration:
            iterator = iter(self.data[dataset])
            x = next(iterator)
        return x

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def mask_txt(self, src_input_idx, idx_to_mask_positions=None, gt_text=None):

        pad_idx = self.params.tokenizer.pad_token_id
        mask_idx = self.params.tokenizer.mask_token_id
        attn = torch.where(src_input_idx != pad_idx, 1, 0)
        attn[:, -1] = 0
        attn[:, 0] = 0
        if idx_to_mask_positions is not None and gt_text is not None:
            attn[idx_to_mask_positions] = 0

        probs_mask_txt = self.params.prob_mask_text * attn
        masked_txt = torch.bernoulli(probs_mask_txt).long()
        idx_to_mask_positions_txt = torch.where(masked_txt)
        gt_text_only = src_input_idx[idx_to_mask_positions_txt].clone()
        src_input_idx.masked_fill_(masked_txt, mask_idx)

        if idx_to_mask_positions is not None and gt_text is not None:
            pos1 = torch.cat((idx_to_mask_positions[0], idx_to_mask_positions_txt[0]))
            pos2 = torch.cat((idx_to_mask_positions[1], idx_to_mask_positions_txt[1]))
            idx_to_mask_positions_txt = (pos1, pos2)

            gt_text_only = torch.cat((gt_text, gt_text_only))

        return src_input_idx, idx_to_mask_positions_txt, gt_text_only

    def mask_txt_from_img_index(self, src_input_idx, reverse_cross_attn):

        # Sample text to mask
        probs_mask_txt = self.params.prob_mask_text * torch.where(reverse_cross_attn.sum(-1) >= 1, 1, 0)
        mask_txt_from_img = torch.bernoulli(probs_mask_txt).long()
        idx_to_mask_positions = torch.where(mask_txt_from_img)
        gt_text = src_input_idx[idx_to_mask_positions].clone()
        src_input_idx.masked_fill_(mask_txt_from_img, self.params.tokenizer.mask_token_id)

        return src_input_idx, idx_to_mask_positions, gt_text

    def get_guided_attn_mask(self, img_labels, src_inps, random_matrix=False):
        global_attn = []
        for bs, im_labs in enumerate(img_labels, 0):
            im_labs = [self.params.tokenizer(im_lab, add_special_tokens=False).input_ids for im_lab in im_labs.split("\n")]
            tensor_attn = []
            tensor = src_inps[bs]
            if not im_labs:
                tensor_attn.append(torch.zeros_like(tensor))
            for lab in im_labs:
                attn = torch.zeros_like(tensor)
                for i in range(tensor.size()[0]):
                    if tensor[i] in lab:
                        attn[i] = 1
                tensor_attn.append(attn)
            global_attn.append(torch.stack(tensor_attn))

        out_attn = pad_sequence(global_attn, batch_first=True)
        if random_matrix:
            prob = 0.25 * torch.ones_like(out_attn)
            out_attn = torch.bernoulli(prob)
        reverse_out_attn = out_attn.permute(0, 2, 1)
        return out_attn, reverse_out_attn

    def preprocessing(self, batch, mode):
        if mode == 'train':
            src_text, tgt_text = batch
        else:
            src_text, tgt_text = batch.values()

        src_inputs = self.params.tokenizer(src_text, return_tensors="pt", padding=True,
                                           max_length=self.params.max_len, truncation=True)
        with self.params.tokenizer.as_target_tokenizer():
            tgt_inputs = self.params.tokenizer(tgt_text, return_tensors="pt", padding=True,
                                               max_length=self.params.max_len, truncation=True)

        src_inps, src_attn = src_inputs.input_ids, src_inputs.attention_mask
        tgt_inps = tgt_inputs.input_ids
        return (src_inps, src_attn), (tgt_inps, None)

    def preprocessing_mmt(self, batch, mode):

        if mode != "commute":
            src_text, tgt_text, clip_features, mdetr_features, boxes_loc, labels = batch.values()
        else:
            src_text, tgt_text, incorrect_tgt_text, clip_features, mdetr_features, boxes_loc, labels = batch.values()

        clip_ft, mdetr_ft = True, True
        if clip_features[0] is None:
            clip_features = None
            clip_ft = False
        if mdetr_features[0] is None:
            mdetr_features = None
            boxes_loc = None
            mdetr_ft = False

        src_inputs = self.params.tokenizer(src_text, return_tensors="pt", padding=True,
                                           max_length=self.params.max_len, truncation=True)
        with self.params.tokenizer.as_target_tokenizer():
            tgt_inputs = self.params.tokenizer(tgt_text, return_tensors="pt", padding=True,
                                               max_length=self.params.max_len, truncation=True)
            if mode == "commute":
                inc_tgt_inputs = self.params.tokenizer(incorrect_tgt_text, return_tensors="pt", padding=True,
                                                       max_length=self.params.max_len, truncation=True)

            src_inps, src_attn = src_inputs.input_ids, src_inputs.attention_mask
            tgt_inps = tgt_inputs.input_ids
            if mode == "commute":
                inc_tgt_inps = inc_tgt_inputs.input_ids

            bs, slen = src_inps.size()
            guided_self_attn_mask, idx_to_mask_positions, gt_text = None, None, None

            if mdetr_ft:
                img_attn = self.get_img_attn_mask(mdetr_features)
            else:
                img_attn = torch.ones((bs, 1))
            if clip_ft and mdetr_ft:
                img_attn = img_attn.unsqueeze(1).repeat(1, img_attn.size(1), 1)
                n_obj = img_attn.size(1)
                img_attn = torch.cat((torch.zeros((bs, 1, n_obj)).long(), img_attn), dim=1)
                img_attn = torch.cat((torch.zeros((bs, n_obj + 1, 1)).long(), img_attn), dim=-1)
                img_attn += torch.eye(n_obj + 1).long().repeat(bs, 1, 1)
                img_attn = img_attn.bool().long()

            if mdetr_ft and self.params.guided_self_attention:
                cross_attn, reverse_cross_attn = self.get_guided_attn_mask(labels, src_inps)
                if clip_ft:
                    cross_attn = torch.cat((src_attn.unsqueeze(1), cross_attn), dim=1)
                    reverse_cross_attn = torch.cat((src_attn.unsqueeze(-1), reverse_cross_attn), dim=-1)
                guided_self_attn_mask = self.build_guided_self_attention(src_attn, cross_attn, reverse_cross_attn, img_attn)

            if len(img_attn.size()) == 3:
                img_attn1d = torch.cat((torch.ones(bs, 1).long(), self.get_img_attn_mask(mdetr_features)), dim=-1)
                i2t_attn = src_attn.unsqueeze(1).repeat(1, img_attn.size(1), 1)
                full_img_attn = torch.cat((img_attn, i2t_attn), dim=-1)
                txt_attn = src_attn.unsqueeze(-1).repeat(1, 1, slen)
                full_txt_attn = torch.cat((img_attn1d.unsqueeze(1).repeat(1, slen, 1), txt_attn), dim=-1)
                src_attn = torch.cat((full_img_attn, full_txt_attn), dim=1)
            else:
                src_attn = torch.cat((img_attn, src_attn), dim=-1)

            if mode != "commute":
                return (src_inps, src_attn), (tgt_inps, None), (clip_features, mdetr_features, boxes_loc, guided_self_attn_mask)
            else:
                return (src_inps, src_attn), (tgt_inps, None), (inc_tgt_inps, None), (
                clip_features, mdetr_features, boxes_loc, guided_self_attn_mask)

    def preprocessing_vmlm(self, batch):

        src_text, clip_features, mdetr_features, boxes_loc, labels = batch.values()

        clip_ft, mdetr_ft = True, True
        if clip_features[0] is None:
            clip_features = None
            clip_ft = False
        if mdetr_features[0] is None:
            mdetr_features = None
            boxes_loc = None
            mdetr_ft = False

        src_inputs = self.params.tokenizer(src_text, return_tensors="pt", padding=True,
                                           max_length=self.params.max_len, truncation=True)

        src_inps, src_attn = src_inputs.input_ids, src_inputs.attention_mask

        bs, slen = src_inps.size()
        guided_self_attn_mask, idx_to_mask_positions, gt_text = None, None, None

        if mdetr_ft:
            img_attn = self.get_img_attn_mask(mdetr_features)
        else:
            img_attn = torch.ones((bs, 1))

        if clip_ft and mdetr_ft:
            img_attn = img_attn.unsqueeze(1).repeat(1, img_attn.size(1), 1)
            n_obj = img_attn.size(1)
            img_attn = torch.cat((torch.zeros((bs, 1, n_obj)).long(), img_attn), dim=1)
            img_attn = torch.cat((torch.zeros((bs, n_obj + 1, 1)).long(), img_attn), dim=-1)
            img_attn += torch.eye(n_obj + 1).long().repeat(bs, 1, 1)
            img_attn = img_attn.bool().long()

        reverse_cross_attn = None
        if mdetr_ft and self.params.guided_self_attention:
            cross_attn, reverse_cross_attn = self.get_guided_attn_mask(labels, src_inps)
            if clip_ft:
                cross_attn = torch.cat((src_attn.unsqueeze(1), cross_attn), dim=1)
                reverse_cross_attn = torch.cat((src_attn.unsqueeze(-1), reverse_cross_attn), dim=-1)
            guided_self_attn_mask = self.build_guided_self_attention(src_attn, cross_attn, reverse_cross_attn, img_attn)

        idx_to_mask_positions, gt_text = None, None
        if mdetr_ft and self.params.prob_mask_text:
            if reverse_cross_attn is None:
                _, reverse_cross_attn = self.get_guided_attn_mask(labels, src_inps)
            src_inps, idx_to_mask_positions, gt_text = self.mask_txt_from_img_index(src_inps,
                                                                                    reverse_cross_attn if not clip_ft else reverse_cross_attn[:, :, 1:])

        if len(img_attn.size()) == 3:
            img_attn1d = torch.cat((torch.ones(bs, 1).long(), self.get_img_attn_mask(mdetr_features)), dim=-1)
            i2t_attn = src_attn.unsqueeze(1).repeat(1, img_attn.size(1), 1)
            full_img_attn = torch.cat((img_attn, i2t_attn), dim=-1)
            txt_attn = src_attn.unsqueeze(-1).repeat(1, 1, slen)
            full_txt_attn = torch.cat((img_attn1d.unsqueeze(1).repeat(1, slen, 1), txt_attn), dim=-1)
            src_attn = torch.cat((full_img_attn, full_txt_attn), dim=1)
        else:
            src_attn = torch.cat((img_attn, src_attn), dim=-1)

        src_inps, idx_to_mask_positions, gt_text = self.mask_txt(src_inps, idx_to_mask_positions, gt_text)

        return (src_inps, src_attn), (idx_to_mask_positions, gt_text), (clip_features, mdetr_features, boxes_loc, guided_self_attn_mask)

    def preprocessing_mlm(self, batch):

        src_text = list(batch.values())[0]
        src_inputs = self.params.tokenizer(src_text, return_tensors="pt", padding=True,
                                           max_length=self.params.max_len, truncation=True)

        src_inps, src_attn = src_inputs.input_ids, src_inputs.attention_mask
        src_inps, idx_to_mask_positions, gt_text = self.mask_txt(src_inps)

        return (src_inps, src_attn), (idx_to_mask_positions, gt_text)

    @staticmethod
    def get_img_attn_mask(img_feats):
        return torch.where((img_feats != 0).sum(-1) != 0, 1, 0)

    @staticmethod
    def build_guided_self_attention(txt_attn, i2t_attn, t2i_attn, img_attn):

        # Format attention mask
        txt_attn_mask = txt_attn.unsqueeze(1).repeat(1, txt_attn.size(1), 1)

        if len(img_attn.size()) == 2:
            img_attn_mask = img_attn.unsqueeze(1).repeat(1, img_attn.size(1), 1)

            # Padding elements attend to nothing except themselves
            identity = torch.eye(img_attn_mask.size(-1)).long().repeat(img_attn_mask.size(0), 1, 1)
            img_attn_mask = torch.where(img_attn_mask + identity == 0, 0, 1)
        else:
            img_attn_mask = img_attn

        # Concat everything
        txt_attn = torch.cat((t2i_attn, txt_attn_mask), dim=-1)
        img_attn = torch.cat((img_attn_mask, i2t_attn), dim=-1)
        attn_mask = torch.cat((img_attn, txt_attn), dim=1)

        return attn_mask

    def mt_step(self, mode="train", batch=None):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params
        model = params.model
        if mode == "train":
            model.train()
        else:
            model.eval()

        lang1_id = params.lang2id[mbart_langs[params.src_lang]]
        lang2_id = params.lang2id[mbart_langs[params.tgt_lang]]

        if batch is None:
            batch = self.get_batch(mode)

        (src_input_idx, src_attention_mask), (tgt_input_idx, _) = self.preprocessing(batch, mode)

        src_input_idx, src_attention_mask, tgt_input_idx = to_cuda(params.device, src_input_idx, src_attention_mask,
                                                                   tgt_input_idx)

        with torch.set_grad_enabled(mode == "train"):
            with autocast():
                outputs = model(input_ids=src_input_idx, attention_mask=src_attention_mask,
                                decoder_input_ids=tgt_input_idx[:, :-1], labels=tgt_input_idx[:, 1:],
                                return_dict=False)

        loss, logits, _ = outputs
        loss /= params.accumulate_gradients

        if params.world_size > 1:
            D.all_reduce(loss, op=D.ReduceOp.SUM)
            loss /= params.world_size

        self.stats[('MT-%s-%s' % (params.src_lang, params.tgt_lang))].append(loss.item() * params.accumulate_gradients)

        if mode == "train":
            # optimize
            self.optimize(loss)

            # number of processed sentences / words
            self.n_sentences += params.batch_size * params.world_size
            self.stats['processed_s'] += params.batch_size * params.world_size

            del src_input_idx, src_attention_mask, tgt_input_idx, logits, loss

        else:
            # Eval perplexity
            try:
                model = params.model.module
            except:
                model = params.model

            model_config = model.config

            pred_idx = tgt_input_idx[:, 1:] != model_config.pad_token_id
            labels = tgt_input_idx[:, 1:].clone()[pred_idx]
            n_words = len(labels)
            xe_loss = loss.item() * n_words

            generated, lengths = None, None
            # Eval BLEU
            if params.eval_bleu:
                length1 = (src_input_idx != model_config.pad_token_id).sum(dim=1)
                max_len = int(1.5 * length1.max().item() + 10)

                generated = model.generate(inputs=src_input_idx, attention_mask=src_attention_mask,
                                           decoder_start_token_id=lang2_id, num_beams=params.beam_size,
                                           length_penalty=params.length_penalty, max_length=max_len,
                                           early_stopping=params.early_stopping, use_cache=True)

            return n_words, xe_loss, (tgt_input_idx, generated, lengths)

    def mmt_step(self, mode="train", batch=None):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params
        model = params.model
        if mode == "train":
            model.train()
        else:
            model.eval()

        lang2_id = params.lang2id[mbart_langs[params.tgt_lang]]

        if batch is None:
            batch = self.get_batch(mode)

        (src_input_idx, src_attention_mask), (tgt_input_idx, _), (clip_features, mdetr_features, boxes_loc, guided_self_attn) \
            = self.preprocessing_mmt(batch, mode)

        text_len = src_input_idx.size(1)
        src_input_idx, src_attention_mask, tgt_input_idx, clip_features, mdetr_features, boxes_loc, guided_self_attn = \
            to_cuda(params.device, src_input_idx, src_attention_mask, tgt_input_idx, clip_features, mdetr_features,
                    boxes_loc, guided_self_attn)

        if clip_features is not None:
            clip_features = clip_features.float()
        img_features = {"global_features": clip_features, "local_features": mdetr_features, "boxes_loc": boxes_loc}
        with torch.set_grad_enabled(mode == "train"):
            with autocast():
                outputs = model(input_ids=src_input_idx,
                                attention_mask=src_attention_mask if guided_self_attn is None else guided_self_attn,
                                decoder_input_ids=tgt_input_idx[:, :-1], labels=tgt_input_idx[:, 1:],
                                img_features=img_features, return_dict=False, text_len=text_len)

        loss, logits, _ = outputs
        loss /= params.accumulate_gradients

        if params.world_size > 1:
            D.all_reduce(loss, op=D.ReduceOp.SUM)
            loss /= params.world_size

        self.stats[('MT-%s-%s' % (params.src_lang, params.tgt_lang))].append(loss.item() * params.accumulate_gradients)

        if mode == "train":
            # optimize
            self.optimize(loss)

            # number of processed sentences / words
            self.n_sentences += params.batch_size * params.world_size
            self.stats['processed_s'] += params.batch_size * params.world_size

            del src_input_idx, src_attention_mask, tgt_input_idx, logits, loss, clip_features, \
                mdetr_features, boxes_loc, guided_self_attn

        else:
            # Eval perplexity
            try:
                model = params.model.module
            except:
                model = params.model

            model_config = model.config
                
            pred_idx = tgt_input_idx[:, 1:] != model_config.pad_token_id
            labels = tgt_input_idx[:, 1:].clone()[pred_idx]
            n_words = len(labels)
            xe_loss = loss.item() * n_words

            generated, lengths = None, None
            # Eval BLEU
            if params.eval_bleu:
                length1 = (src_input_idx != model_config.pad_token_id).sum(dim=1)
                max_len = int(1.5 * length1.max().item() + 10)

                model_kwargs = {"img_features": img_features, "text_len": text_len}
                generated = model.generate(inputs=src_input_idx, attention_mask=src_attention_mask if guided_self_attn is None else guided_self_attn,
                                           decoder_start_token_id=lang2_id, num_beams=params.beam_size,
                                           length_penalty=params.length_penalty, max_length=max_len,
                                           early_stopping=params.early_stopping, use_cache=True, **model_kwargs)

            return n_words, xe_loss, (tgt_input_idx, generated, lengths)

    def vmlm_step(self, mode="train", batch=None):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params
        model = params.model
        if mode == "train" or mode == "train_mix":
            model.train()
        else:
            model.eval()

        if batch is None:
            batch = self.get_batch(mode)

        if mode == "train_mix":
            mode = "train"

        (src_input_idx, src_attention_mask), (idx_to_mask_positions, gt_text), (clip_features, mdetr_features, boxes_loc, guided_self_attn) \
            = self.preprocessing_vmlm(batch)

        tgt_input_idx = src_input_idx.clone()
        tgt_input_idx[idx_to_mask_positions] = gt_text
        text_len = src_input_idx.size(1)

        src_input_idx, src_attention_mask, clip_features, mdetr_features, boxes_loc, guided_self_attn, gt_text, tgt_input_idx = \
            to_cuda(params.device, src_input_idx, src_attention_mask, clip_features, mdetr_features, boxes_loc,
                    guided_self_attn, gt_text, tgt_input_idx)

        if clip_features is not None:
            clip_features = clip_features.float()
        img_features = {"global_features": clip_features, "local_features": mdetr_features, "boxes_loc": boxes_loc}
        with torch.set_grad_enabled(mode == "train"):
            with autocast():
                loss, logits, _ = model(input_ids=src_input_idx,
                                        attention_mask=src_attention_mask if guided_self_attn is None else guided_self_attn,
                                        decoder_input_ids=tgt_input_idx[:, :-1], labels=gt_text, pred_idx=idx_to_mask_positions,
                                        img_features=img_features, return_dict=False, text_len=text_len)

        loss /= params.accumulate_gradients

        if params.world_size > 1:
            D.all_reduce(loss, op=D.ReduceOp.SUM)
            loss /= params.world_size

        # Accuracy
        num_preds = gt_text.size(0)
        preds = logits.argmax(-1)
        acc = (preds == gt_text).sum()
        if params.world_size > 1:
            acc = gather_tensor(acc.unsqueeze(0)).sum()
            num_preds = gather_tensor(torch.Tensor([num_preds]).to(params.device)).sum()
            acc = acc.float() / num_preds.float()
            acc, num_preds = acc.item(), num_preds.item()
        else:
            acc = acc.item() / num_preds

        self.stats[('VMLM-loss-%s' % params.src_lang)].append(loss.item() * params.accumulate_gradients)
        self.stats[('VMLM-acc-%s' % params.src_lang)].append(acc)

        if mode == "train":
            # optimize
            self.optimize(loss)

            # number of processed sentences / words
            self.n_sentences += params.batch_size * params.world_size
            self.stats['processed_s'] += params.batch_size * params.world_size

            del src_input_idx, src_attention_mask, idx_to_mask_positions, gt_text, logits, loss, clip_features, \
                mdetr_features, boxes_loc, guided_self_attn

        else:
            return loss, acc, num_preds

    def mlm_step(self, mode="train", batch=None):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """

        params = self.params
        model = params.model
        if mode == "train" or mode == "train_mix":
            model.train()
        else:
            model.eval()

        if batch is None:
            batch = self.get_batch(mode)

        if mode == "train_mix":
            mode = "train"

        (src_input_idx, src_attention_mask), (idx_to_mask_positions, gt_text) = self.preprocessing_mlm(batch)

        tgt_input_idx = src_input_idx.clone()
        tgt_input_idx[idx_to_mask_positions] = gt_text

        src_input_idx, src_attention_mask, gt_text = to_cuda(params.device, src_input_idx, src_attention_mask, gt_text)

        with torch.set_grad_enabled(mode == "train"):
            with autocast():
                loss, logits, _ = model(input_ids=src_input_idx,
                                        attention_mask=src_attention_mask,
                                        decoder_input_ids=tgt_input_idx[:, :-1], labels=gt_text,
                                        pred_idx=idx_to_mask_positions,
                                        return_dict=False)

        loss /= params.accumulate_gradients

        if params.world_size > 1:
            D.all_reduce(loss, op=D.ReduceOp.SUM)
            loss /= params.world_size

        # Accuracy
        num_preds = gt_text.size(0)
        preds = logits.argmax(-1)
        acc = (preds == gt_text).sum()
        if params.world_size > 1:
            acc = gather_tensor(acc.unsqueeze(0)).sum()
            num_preds = gather_tensor(torch.Tensor([num_preds]).to(params.device)).sum()
            acc = acc.float() / num_preds.float()
            acc, num_preds = acc.item(), num_preds.item()
        else:
            acc = acc.item() / num_preds

        self.stats[('MLM-loss-%s' % params.src_lang)].append(loss.item() * params.accumulate_gradients)
        self.stats[('MLM-acc-%s' % params.src_lang)].append(acc)

        if mode == "train":
            # optimize
            self.optimize(loss)

            # number of processed sentences / words
            self.n_sentences += params.batch_size * params.world_size
            self.stats['processed_s'] += params.batch_size * params.world_size

            del src_input_idx, src_attention_mask, idx_to_mask_positions, gt_text, logits, loss

        else:
            return loss, acc, num_preds

    def evaluate_mt(self, mode="test"):  # TODO Program eval function

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})
        try:
            model_config = params.model.config
        except:
            model_config = params.model.module.config

        xe_loss = 0
        n_words = 0
        # store hypothesis to compute BLEU score
        if params.eval_bleu:
            refs, hypothesis = [], []

        for idx, batch in enumerate(self.data[mode]):

            n_word, x_loss, (tgt_input_idx, generated, lengths) = self.mt_step(mode, batch)

            n_words += n_word
            xe_loss += x_loss

            if self.params.world_size > 1:
                tgt_input_idx = gather_tensor(
                    pad_tensor(tgt_input_idx, max_len=params.max_len, pad_idx=model_config.pad_token_id))
                generated = gather_tensor(
                    pad_tensor(generated, max_len=params.max_len, pad_idx=model_config.pad_token_id))

            if params.eval_bleu:
                refs += params.tokenizer.batch_decode(tgt_input_idx, skip_special_tokens=True)
                hypothesis += params.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # compute perplexity
        scores['%s_%s-%s_mt_ppl' % (mode, params.src_lang, params.tgt_lang)] = np.exp(xe_loss / n_words)

        # compute BLEU
        if params.eval_bleu and params.global_rank <= 0:
            # hypothesis / reference paths
            ref_data = params.test_data_set if params.test_data_set else mode
            if params.eval_only and not params.test_data_set:
                ref_data = "test_2016_flickr"
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], params.src_lang, params.tgt_lang, ref_data)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            init_epoch = str(params.init_epoch) if params.init_epoch >= 0 else str(0)
            ref_name = 'ref.{0}-{1}.{2}.iter_{3}.txt'.format(params.src_lang, params.tgt_lang, ref_data, init_epoch)
            ref_path = os.path.join(params.hyp_path, ref_name)
            if not os.path.exists(ref_path):
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(refs) + '\n')

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (mode, params.src_lang, params.tgt_lang)] = bleu

        return scores

    def evaluate_mmt(self, mode="test"):

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})
        try:
            model_config = params.model.config
        except:
            model_config = params.model.module.config

        xe_loss = 0
        n_words = 0
        # store hypothesis to compute BLEU score
        if params.eval_bleu:
            refs, hypothesis = [], []

        for idx, batch in enumerate(self.data[mode]):

            n_word, x_loss, (tgt_input_idx, generated, lengths) = self.mmt_step(mode, batch)

            n_words += n_word
            xe_loss += x_loss

            if self.params.world_size > 1:
                tgt_input_idx = gather_tensor(
                    pad_tensor(tgt_input_idx, max_len=params.max_len, pad_idx=model_config.pad_token_id))
                generated = gather_tensor(
                    pad_tensor(generated, max_len=params.max_len, pad_idx=model_config.pad_token_id))

            if params.eval_bleu:
                refs += params.tokenizer.batch_decode(tgt_input_idx, skip_special_tokens=True)
                hypothesis += params.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # compute perplexity
        scores['%s_%s-%s_mt_ppl' % (mode, params.src_lang, params.tgt_lang)] = np.exp(xe_loss / n_words)

        # compute BLEU
        if params.eval_bleu and params.global_rank <= 0:
            # hypothesis / reference paths
            ref_data = params.test_data_set if params.test_data_set else mode
            if params.eval_only and not params.test_data_set:
                ref_data = "test_2016_flickr"
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], params.src_lang, params.tgt_lang, ref_data)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            init_epoch = str(params.init_epoch) if params.init_epoch >= 0 else str(0)
            ref_name = 'ref.{0}-{1}.{2}.iter_{3}.txt'.format(params.src_lang, params.tgt_lang, ref_data, init_epoch)
            ref_path = os.path.join(params.hyp_path, ref_name)
            if not os.path.exists(ref_path):
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(refs) + '\n')

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (mode, params.src_lang, params.tgt_lang)] = bleu

        return scores

    def evaluate_vmlm(self, mode="valid"):

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})

        tot_acc = 0
        tot_pred = 0

        for idx, batch in enumerate(self.data[mode]):

            _, acc, num_preds = self.vmlm_step(mode, batch)

            tot_acc += acc * num_preds
            tot_pred += num_preds

        # compute loss & accuracy
        scores['%s_%s_vmlm_acc' % (mode, params.src_lang)] = tot_acc / tot_pred

        return scores

    def evaluate_mlm(self, mode="valid"):

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})

        tot_acc = 0
        tot_pred = 0

        for idx, batch in enumerate(self.data[mode]):
            
            _, acc, num_preds = self.mlm_step(mode, batch)

            tot_acc += acc * num_preds
            tot_pred += num_preds

        # compute loss & accuracy
        scores['%s_%s_mlm_acc' % (mode, params.src_lang)] = tot_acc / tot_pred

        return scores

    def evaluate_commute(self):

        params = self.params
        model = params.model
        model.eval()
        scores = OrderedDict({'epoch': self.epoch})

        cor_ppl, inc_ppl, global_scores = [], [], []

        for idx, batch in tqdm(enumerate(self.data["test"])):

            (src_input_idx, src_attention_mask), (tgt_input_idx, _), (incorrect_input_idx, _), (
            clip_features, mdetr_features, boxes_loc, guided_self_attn) \
                = self.preprocessing_mmt(batch, "commute")
            text_len = src_input_idx.size(1)

            src_input_idx, src_attention_mask, tgt_input_idx, incorrect_input_idx, clip_features, mdetr_features, \
            boxes_loc, guided_self_attn = \
                to_cuda(params.device, src_input_idx, src_attention_mask, tgt_input_idx, incorrect_input_idx,
                        clip_features, mdetr_features, boxes_loc, guided_self_attn)

            img_features = {"global_features": clip_features, "local_features": mdetr_features, "boxes_loc": boxes_loc}

            with torch.no_grad():
                with autocast():
                    outputs = model(input_ids=src_input_idx,
                                    attention_mask=src_attention_mask if guided_self_attn is None else guided_self_attn,
                                    decoder_input_ids=tgt_input_idx[:, :-1], labels=tgt_input_idx[:, 1:],
                                    img_features=img_features, return_dict=False, text_len=text_len)

                    inc_outputs = model(input_ids=src_input_idx,
                                        attention_mask=src_attention_mask if guided_self_attn is None else guided_self_attn,
                                        decoder_input_ids=incorrect_input_idx[:, :-1], labels=incorrect_input_idx[:, 1:],
                                        img_features=img_features, return_dict=False, text_len=text_len)

            loss, logits, _ = outputs
            inc_loss, inc_logits, _ = inc_outputs

            # Eval perplexity
            cor_xe_loss = loss.item()
            inc_xe_loss = inc_loss.item()

            cor_ppl.append(np.exp(cor_xe_loss))
            inc_ppl.append(np.exp(inc_xe_loss))
            global_scores.append(1 if cor_ppl[-1] <= inc_ppl[-1] else 0)

        split = "_" + params.test_data_set.split("_")[-1] if len(params.test_data_set.split("_")) == 2 else ""
        f_cor_save, f_inc_save = f"commute{split}.correct_ppl.{params.tgt_lang}", f"commute{split}.incorrect_ppl.{params.tgt_lang}"
        f_score_save = f"commute{split}.accuracy_commute.fr"

        with open(os.path.join(params.hyp_path, f_cor_save), "w") as f:
            f.write("\n".join([str(s) for s in cor_ppl]))

        with open(os.path.join(params.hyp_path, f_inc_save), "w") as f:
            f.write("\n".join([str(s) for s in inc_ppl]))

        with open(os.path.join(params.hyp_path, f_score_save), "w") as f:
            f.write("\n".join([str(s) for s in global_scores]))

        with open(os.path.join(os.path.dirname(params.hyp_path), f"results_CoMMuTE{split}.log"), "w") as f:
            f.write(str(np.mean(global_scores)))

        return scores

    def evaluate_commute_generation(self):

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})
        try:
            model_config = params.model.config
        except:
            model_config = params.model.module.config

        xe_loss = 0
        n_words = 0
        # store hypothesis to compute BLEU score
        if params.eval_bleu:
            refs, hypothesis = [], []

        for idx, batch in enumerate(self.data["test"]):

            n_word, x_loss, (tgt_input_idx, generated, lengths) = self.mmt_step("test", batch)

            n_words += n_word
            xe_loss += x_loss

            if self.params.world_size > 1:
                tgt_input_idx = gather_tensor(
                    pad_tensor(tgt_input_idx, max_len=params.max_len, pad_idx=model_config.pad_token_id))
                generated = gather_tensor(
                    pad_tensor(generated, max_len=params.max_len, pad_idx=model_config.pad_token_id))

            if params.eval_bleu:
                refs += params.tokenizer.batch_decode(tgt_input_idx, skip_special_tokens=True)
                hypothesis += params.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # compute perplexity
        scores['%s_%s-%s_mt_ppl' % ("test", params.src_lang, params.tgt_lang)] = np.exp(xe_loss / n_words)

        # compute BLEU
        if params.eval_bleu and params.global_rank <= 0:
            # hypothesis / reference paths
            ref_data = "commute_generation"
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], params.src_lang, params.tgt_lang, ref_data)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            init_epoch = str(params.init_epoch) if params.init_epoch >= 0 else str(0)
            ref_name = 'ref.{0}-{1}.{2}.iter_{3}.txt'.format(params.src_lang, params.tgt_lang, ref_data, init_epoch)
            ref_path = os.path.join(params.hyp_path, ref_name)
            if not os.path.exists(ref_path):
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(refs) + '\n')

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % ("test", params.src_lang, params.tgt_lang)] = bleu

        return scores

    def evaluate_commute_mt_generation(self):

        params = self.params
        scores = OrderedDict({'epoch': self.epoch})
        try:
            model_config = params.model.config
        except:
            model_config = params.model.module.config

        xe_loss = 0
        n_words = 0
        # store hypothesis to compute BLEU score
        if params.eval_bleu:
            refs, hypothesis = [], []

        for idx, batch in enumerate(self.data["test"]):

            n_word, x_loss, (tgt_input_idx, generated, lengths) = self.mt_step("test", batch)

            n_words += n_word
            xe_loss += x_loss

            if self.params.world_size > 1:
                tgt_input_idx = gather_tensor(
                    pad_tensor(tgt_input_idx, max_len=params.max_len, pad_idx=model_config.pad_token_id))
                generated = gather_tensor(
                    pad_tensor(generated, max_len=params.max_len, pad_idx=model_config.pad_token_id))

            if params.eval_bleu:
                refs += params.tokenizer.batch_decode(tgt_input_idx, skip_special_tokens=True)
                hypothesis += params.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # compute perplexity
        scores['%s_%s-%s_mt_ppl' % ("test", params.src_lang, params.tgt_lang)] = np.exp(xe_loss / n_words)

        # compute BLEU
        if params.eval_bleu and params.global_rank <= 0:
            # hypothesis / reference paths
            ref_data = "commute_generation"
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], params.src_lang, params.tgt_lang, ref_data)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            init_epoch = str(params.init_epoch) if params.init_epoch >= 0 else str(0)
            ref_name = 'ref.{0}-{1}.{2}.iter_{3}.txt'.format(params.src_lang, params.tgt_lang, ref_data, init_epoch)
            ref_path = os.path.join(params.hyp_path, ref_name)
            if not os.path.exists(ref_path):
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(refs) + '\n')

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % ("test", params.src_lang, params.tgt_lang)] = bleu

        return scores
