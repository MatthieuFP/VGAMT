import os
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from transformers.adapters import AdapterConfig
from .vgamt import VGAMT

from logging import getLogger

logger = getLogger()


xlm_model = "xlm-mlm-100-1280"
mbart_model = "facebook/mbart-large-50"
mbart_langs = {'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'fr': 'fr_XX'}


def load_pretrained_model(params):
    model = MBartForConditionalGeneration.from_pretrained(os.path.join(params.cache_dir, mbart_model))
    tokenizer = MBart50Tokenizer.from_pretrained(os.path.join(params.cache_dir, mbart_model),
                                                 src_lang=mbart_langs[params.src_lang],
                                                 tgt_lang=mbart_langs[params.tgt_lang])
    return model, tokenizer


def build_model(params):
    model, tokenizer = load_pretrained_model(params)
    params.config = model.config
    params.config.update({"label_smoothing": params.smoothing})
    if not params.multimodal_model and params.adapters:
        model = MBartForConditionalGenerationWithAdapters(params.config)
    elif params.multimodal_model:
        model = VGAMT(params.config, params)
    return model, tokenizer


def reload_model(params):
    logger.info("Reloading model from %s ..." % params.reload_model)
    checkpoint = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))
    reloaded = checkpoint['model']
    if all([k.startswith('module.') for k in reloaded.keys()]):
        reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

    miss, unexp = params.model.load_state_dict(reloaded, strict=False)
    for name in sorted(miss):
        logger.info(f'Model reloading: missing parameter {name} will be randomly initialized')
    for name in sorted(unexp):
        logger.info(f'Model reloading: unexpected parameter {name} ignored')

    if not params.start_new_xp_from_ckpt:
        params.init_epoch = checkpoint["epoch"]
        params.n_total_iter = checkpoint["n_total_iter"]
    else:
        params.init_epoch = 0
        params.n_total_iter = 0

    if params.reload_optim:
        params.optim_from_ckpt = checkpoint["optimizer"]
        logger.info("Optimizer reloaded with success !")

    return None


class MBartForConditionalGenerationWithAdapters(MBartForConditionalGeneration):
    """
    Text-only Machine translation model fine-tuned from mBART
    """
    def __init__(self, config):
        super(MBartForConditionalGenerationWithAdapters, self).__init__(config)

        self.adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=8,
                                            non_linearity="relu")
        self.model.add_adapter("bottleneck_adapter", config=self.adapter_config)
        self.model.set_active_adapters("bottleneck_adapter")

        self.freeze_text_params()

    def freeze_text_params(self):
        for name, param in self.model.named_parameters():
            if "adapters" not in name:
                param.requires_grad = False

