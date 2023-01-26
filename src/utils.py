import os
import sys
import pickle
import random
import argparse
import subprocess
import torch
import torch.distributed as D
from torch._six import string_classes
import sacrebleu
import collections
from .logger import create_logger
import hostlist
import re
from torch.nn.utils.rnn import pad_sequence


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def get_labels(decoder_inputs: torch.Tensor):
    """
    Shift input ids one token to the left to get labels from decoder_inputs.
    """
    shifted_input_ids = decoder_inputs.new_zeros(decoder_inputs.shape)
    shifted_input_ids[:, :-1] = decoder_inputs[:, 1:].clone()

    return shifted_input_ids[:, :-1]


def to_cuda(gpu, *args):
    """
    Move tensors to CUDA.
    """
    if torch.cuda.is_available():
        return [None if x is None else x.to(gpu) for x in args]  # non-blocking=True
    else:
        return[None if x is None else x for x in args]


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    if (params.reload_model != '' or params.eval_only) and not params.start_new_xp_from_ckpt:
        params.dump_path = os.path.dirname(params.reload_model)
        epoch = re.sub("[^0-9]", "", os.path.basename(params.reload_model))
    else:
        get_dump_path(params)
        pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    hyp_dir = "hypothesis_" + params.test_data_set if params.test_data_set else "hypothesis"
    if "commute" in params.test_data_set and params.commute_generation:
        hyp_dir += "_generation"
    params.hyp_path = os.path.join(params.dump_path, hyp_dir)
    os.makedirs(params.hyp_path, exist_ok=True)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    if params.reload_model != '' and not params.eval_only and not params.start_new_xp_from_ckpt:
        logger_name = f'train_epoch{epoch}.log'
    elif params.reload_model != '' and params.eval_only:
        logger_name = params.test_data_set + ".log" if params.test_data_set else "test.log"
        if "commute" in params.test_data_set and params.commute_generation:
            logger_name = logger_name.split(".")[0] + "_generation.log"
    else:
        logger_name = 'train.log'
    logger = create_logger(os.path.join(params.dump_path, logger_name),
                           rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    dump_path = "./dummy_models" if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')

    hyps, refs = [], []

    with open(hyp) as fh, open(ref) as rh:
        for line in fh:
            hyps.append(line.strip())

        for line in rh:
            refs.append(line.strip())

        score = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none').score

    return score


def init_env_variables(params):

    # get SLURM variables
    params.rank = int(os.environ['SLURM_PROCID'])
    params.local_rank = int(os.environ['SLURM_LOCALID'])
    params.size = int(os.environ['SLURM_NTASKS'])
    params.cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    # get node list from slurm
    params.hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    # get IDs of reserved GPU
    params.gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = params.hostnames[0]
    os.environ['MASTER_PORT'] = str(12345 + int(min(params.gpu_ids)))  # to avoid port conflict on the same node

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


def pad_tensor(x, max_len, pad_idx):
    bs, len = x.size()
    if len > max_len:
        x = x[:, :max_len]
    device = x.device
    tensor = torch.zeros([bs, max_len], dtype=torch.long, device=device)
    tensor[:, :len] = x
    tensor[:, len:] = pad_idx
    return tensor


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            padded_tensor = pad_sequence([torch.as_tensor(b) for b in batch], batch_first=True, padding_value=0.)
            return custom_collate([padded_tensor[idx] for idx, b in enumerate(batch)])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) if elem[key] is not None else [None] * len(batch) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
