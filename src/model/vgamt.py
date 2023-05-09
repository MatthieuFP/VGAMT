import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBartModel, MBartConfig
from transformers.adapters import AdapterConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput
from transformers.models.mbart.modeling_mbart import _expand_mask, shift_tokens_right, MBartEncoder
from logging import getLogger
from typing import List, Optional, Tuple, Union


logger = getLogger()

mbart_model = "facebook/mbart-large-50"
mbart_langs = {'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'fr': 'fr_XX'}


class ImgProjector(nn.Module):
    """Image feature projection layer."""
    def __init__(self, config, type="full"):
        super().__init__()
        inp_dim = 512 if type == "full" else 64
        self.linear = nn.Linear(inp_dim, config.d_model)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)


class RegionalEncodings(nn.Module):
    """Bounding-box projection for positional encodings of images."""
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(4, config.d_model)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)


class MultimodalTransformerEncoder(MBartEncoder):
    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super(MultimodalTransformerEncoder, self).__init__(config, embed_tokens)

        self.proj_full_img = ImgProjector(config, "full")
        self.proj_reg_img = ImgProjector(config, "reg")
        self.region_encoder = RegionalEncodings(config)
        self.img_embedding = nn.Linear(1, config.d_model)

        # Layer Norms
        self.layer_norm_full_img = nn.LayerNorm(config.d_model)
        self.layer_norm_reg_img = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        img_features: Optional[dict] = None,
        text_len: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`MBartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        img_feats = tuple()
        # Add visual features
        if img_features["global_features"] is not None:
            full_img_embeds = self.proj_full_img(img_features["global_features"])
            full_img_embeds = self.layer_norm_full_img(full_img_embeds)
            full_img_embeds += self.img_embedding.weight.t()
            img_feats = img_feats + (full_img_embeds.unsqueeze(1),)

        if img_features["local_features"] is not None:
            assert img_features["boxes_loc"] is not None
            reg_img_embeds = self.proj_reg_img(img_features["local_features"])
            reg_img_embeds = self.layer_norm_reg_img(reg_img_embeds)
            reg_img_embeds += self.img_embedding.weight.t().unsqueeze(0)
            reg_img_embeds += self.region_encoder(img_features["boxes_loc"])
            img_feats = img_feats + (reg_img_embeds,)

        if len(img_feats):
            all_feats = img_feats + (hidden_states,)
            hidden_states = torch.cat(all_feats, dim=1)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            if len(attention_mask.size()) == 2:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
            else:
                attention_mask = attention_mask.unsqueeze(1)
                attention_mask = 1.0 - attention_mask
                attention_mask.masked_fill_(attention_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MultimodalTransformerModel(MBartModel):
    def __init__(self, config, params):
        super(MultimodalTransformerModel, self).__init__(config)

        self.params = params
        self.encoder = MultimodalTransformerEncoder(config, self.shared)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            img_features: Optional[dict] = None,
            text_len: Optional[int] = None,
    ) -> Union[Seq2SeqModelOutput, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_features=img_features,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        ## TODO: define encoder_attn_mask_text_only option somewhere
        if self.params.encoder_attn_mask_text_only:
            text_len = input_ids.size(1) if input_ids is not None else text_len
            encoder_hidden_states = encoder_outputs[0][:, -text_len:]
            if len(attention_mask.size()) == 3:  # if guided self-attention for instance, get the block text only in the attention masks.
                attention_mask = attention_mask[:, -text_len:, -text_len:][:, 0, :]
            else:
                attention_mask = attention_mask[:, -text_len:]
        else:
            encoder_hidden_states = encoder_outputs[0]
            ## TODO: custom attention mask based on img padding AND text padding

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VGAMT(MBartForConditionalGeneration):
    def __init__(self, config, params):
        super(VGAMT, self).__init__(config)

        self.model = MultimodalTransformerModel(config, params)
        if params.adapters:
            self.adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=8,
                                                non_linearity="relu")
            self.model.add_adapter("bottleneck_adapter", config=self.adapter_config)
            self.model.set_active_adapters("bottleneck_adapter")

        if params.freeze_text_parameters:
            self.freeze_text_params()

    def freeze_text_params(self):
        for name, param in self.model.named_parameters():
            if ("adapters" not in name) and ("proj_full_img" not in name) and ("proj_reg_img" not in name) and ("region_encoder" not in name):
                param.requires_grad = False

    def freeze_encoder(self):
        for name, params in self.model.encoder.named_parameters():
            params.requires_grad = False

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            img_features: Optional[dict] = None,
            text_len: Optional[int] = None,
            pred_idx: Optional[tuple] = None,
        ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

       	"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_features=img_features,
                text_len=text_len,
            )

        lm_logits = self.model.encoder.invertible_adapters_forward(outputs[0], rev=True)
        lm_logits = self.lm_head(lm_logits) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            if pred_idx is None:
                pred_idx = labels != self.config.pad_token_id
                lm_logits = lm_logits[pred_idx.unsqueeze(-1).expand_as(lm_logits)].view(-1, lm_logits.size(-1))
                labels = labels[pred_idx]
            else:
                pred_pos = pred_idx[0], pred_idx[1] - 1
                lm_logits = lm_logits[pred_pos]
                assert lm_logits.size(0) == labels.size(0), "Labels shape must match lm_logits size"

            if self.config.label_smoothing:
                smooth_ratio = self.config.label_smoothing
                confidence = 1 - smooth_ratio
                smoothing_value = smooth_ratio / (self.config.vocab_size - 2)
                one_hot = torch.full((self.config.vocab_size,), smoothing_value)
                one_hot[self.config.pad_token_id] = 0
                one_hot = one_hot.unsqueeze(0).cuda()

                loss_fct = nn.KLDivLoss(reduction="sum")
                truth_p = one_hot.repeat(labels.size(0), 1)
                truth_p.scatter_(1, labels.unsqueeze(1), confidence)
                truth_p.masked_fill_((labels == self.config.pad_token_id).unsqueeze(1), 0)
                masked_lm_loss = loss_fct(torch.log_softmax(lm_logits.view(-1, self.config.vocab_size), dim=-1), truth_p) / labels.size(0)
            else:
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "img_features": kwargs["img_features"] if "img_features" in kwargs.keys() else None,
            "text_len": kwargs["text_len"] if "text_len" in kwargs.keys() else None,
        }


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super(PredLayer, self).__init__()

        self.config = config
        dim = self.config.d_model

        # self.layer_norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.config.vocab_size, bias=True)

    def forward(self, x, labels):
        """
        Compute the loss, and optionally the scores.
        """

        if self.config.label_smoothing:
            confidence = 1 - self.config.label_smoothing
            smoothing_value = self.config.label_smoothing / (self.config.vocab_size - 2)
            one_hot = torch.full((self.config.vocab_size,), smoothing_value)
            one_hot[self.config.pad_token_id] = 0
            one_hot = one_hot.unsqueeze(0).cuda()

            criterion = nn.KLDivLoss(reduction="sum")

        scores = self.proj(x)  # self.proj(self.layer_norm(x))
        # scores = out.view(-1, self.config.vocab_size)

        if self.config.label_smoothing:
            truth_p = one_hot.repeat(labels.size(0), 1)
            truth_p.scatter_(1, labels.unsqueeze(1), confidence)
            truth_p.masked_fill_((labels == self.config.pad_token_id).unsqueeze(1), 0)
            loss = criterion(torch.log_softmax(scores, dim=-1), truth_p) / labels.size(0)  # Scale the loss
        else:
            loss = F.cross_entropy(scores, labels, reduction='mean')

        return scores, loss
