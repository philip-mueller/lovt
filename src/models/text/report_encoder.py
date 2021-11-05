import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Any, Dict

import torch
from omegaconf import MISSING
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from models.components.aggregation import get_aggregator
from models.components.utils import EncoderInterface, AttentionMask, EncoderOutput, EncoderConfig
from models.text.language_model_loader import load_language_model
from models.text.report_transforms import ReportAugmentationConfig, ReportDataTransform
from common.dataclass_utils import TensorDataclassMixin

log = logging.getLogger(__name__)


@dataclass
class ReportEncoderConfig(EncoderConfig):
    _encoder_cls_: str = 'ReportEncoderModel'
    modality: str = 'report'

    language_encoder: str = MISSING
    max_length: int = MISSING
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 0

    # options: 'max', 'avg', 'token_0', 'learned_query_attention', 'tanh_self_attention'
    global_aggregator: str = 'max'  # ConVIRT: 'max
    global_from_local: bool = False
    local_weights_from_aggregator: bool = False

    # options: 'tokens' or 'sentences'
    local_from: str = 'tokens'
    # joint: encode all sentences concatenated and split afterwards, independent: encode each sentence on its own
    sentence_encoding_mode: str = 'joint'
    # options: 'max', 'avg', 'learned_query_attention', 'tanh_self_attention', 'token_0
    sentence_aggregator: str = 'max'

    project_language_embeddings: bool = False
    d: Optional[int] = None  # 96

    data_augmentation: ReportAugmentationConfig = ReportAugmentationConfig()


class ReportEncoderModel(EncoderInterface, nn.Module):
    def __init__(self, config: ReportEncoderConfig):
        super(ReportEncoderModel, self).__init__()
        self.config = config
        self.concatenate_sentences = True

        log.info(f'Instantiating ReportEncoderModel with language model {config.language_encoder}')
        self.language_model, self.tokenizer = load_language_model(config.language_encoder)
        if config.freeze_embeddings:
            for param in self.language_model.embeddings.parameters():
                param.requires_grad = False
        for layer_index, layer in enumerate(self.language_model.encoder.layer):
            if layer_index < config.freeze_encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        log.info(f'Language model {type(self.language_model)} '
                 f'with {config.freeze_encoder_layers}/{len(self.language_model.encoder.layer)} frozen layers '
                 f'and freeze_embeddings={config.freeze_embeddings}')
        self.max_length = self.config.max_length

        if self.config.sentence_encoding_mode == 'joint':
            self.batch_collator = ReportJointSentenceBatchCollator(self.tokenizer, self.max_length)
        elif self.config.sentence_encoding_mode == 'independent':
            self.batch_collator = ReportIndependentSentenceBatchCollator(self.tokenizer, self.max_length)
        else:
            raise ValueError(self.config.sentence_encoding_mode)

        self.d_language_model = self.language_model.config.hidden_size

        if config.d is None:
            config.d = self.d_language_model
        if not config.project_language_embeddings:
            assert config.d == self.d_language_model

        # global and local are currently the same (global is only aggregated local)
        self.d_l = config.d
        self.d_g = config.d

        self.input_projection = nn.Linear(self.d_language_model, config.d, bias=False) \
            if config.project_language_embeddings else None
        self.global_aggregator = get_aggregator(config.global_aggregator, d=config.d)
        log.info(f'Global aggregator {type(self.global_aggregator)}')

        if self.input_projection:
            nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_out', nonlinearity='linear')

        assert config.local_from in ('tokens', 'sentences')
        if config.local_from == 'sentences':
            self.local_from_sentences = True
            self.sentence_aggregator = get_aggregator(config.sentence_aggregator, d=config.d)
            log.info(f'Using sentences for local with sentence aggregator {type(self.sentence_aggregator)}')
        else:
            self.local_from_sentences = False
            log.info(f'Using tokens for local')

        self.transform = ReportDataTransform(config.data_augmentation)
        self.val_transform = ReportDataTransform(config.data_augmentation, val=True)

    def update_data_augmentation(self, data_augmentation_config: Optional[Any] = None, dataset_stats=None):
        if not data_augmentation_config:
            data_augmentation_config = ReportAugmentationConfig(augment=False)
        elif isinstance(data_augmentation_config, bool) and data_augmentation_config:
            data_augmentation_config = ReportAugmentationConfig()
        elif not isinstance(data_augmentation_config, ReportAugmentationConfig):
            data_augmentation_config = ReportAugmentationConfig(**data_augmentation_config)

        self.config.data_augmentation = data_augmentation_config
        self.transform = ReportDataTransform(data_augmentation_config)
        self.val_transform = ReportDataTransform(data_augmentation_config, val=True)

    @property
    def max_region_size(self):
        return self.max_length

    def forward(self,
                input_ids: torch.Tensor,
                sentence_splits: 'BatchSentenceSplitsInfo',
                attention_mask: Optional[Union['AttentionMask', torch.Tensor]],
                special_tokens_mask: torch.Tensor,
                return_local=True, return_global=True, **kwargs):
        """
        Note: input_ids, sentence_splits, attention_mask, and special_tokens_mask come from the collate method.
        :param input_ids: (B x N)
        :param sentence_splits:
        :param attention_mask: (B x N)
        :param special_tokens_mask: (B x N)
        :param return_local:
        :param return_global:
        :param kwargs:
        :return:
        """
        # (B x N)
        attention_mask = AttentionMask.from_binary_mask_or_attention_mask(attention_mask,
                                                                          dtype=self.language_model.dtype)
        encoded: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.language_model(input_ids=input_ids, attention_mask=attention_mask.binary_mask, **kwargs)

        if self.config.sentence_encoding_mode == 'joint':
            return self.encode_joint(encoded, attention_mask, sentence_splits,
                                     special_tokens_mask, return_global, return_local)
        elif self.config.sentence_encoding_mode == 'independent':
            return self.encode_independent(encoded, attention_mask, sentence_splits,
                                           special_tokens_mask, return_global, return_local)
        else:
            raise ValueError(self.config.sentence_encoding_mode)

    def encode_independent(self, encoded, attention_mask, sentence_splits,
                           special_tokens_mask, return_global, return_local):
        # (N_batch_sent x N_sent_tok x d_language_model)
        all_token_features = encoded.last_hidden_state
        if self.input_projection:
            # (N_batch_sent x N_sent_tok x d_l)
            all_token_features = self.input_projection(all_token_features)

        # (B x N_sent x N_sent_tok x d_l)
        sentence_token_features, sentence_mask, sentence_token_mask = \
            restructure_sentences_into_batch(all_token_features, sentence_splits)
        sentence_mask = AttentionMask.from_binary_mask_or_attention_mask(sentence_mask,
                                                                         dtype=all_token_features.dtype)
        sentence_token_mask = AttentionMask.from_binary_mask_or_attention_mask(sentence_token_mask,
                                                                               dtype=all_token_features.dtype)
        B, N_sent, N_sent_tok, d_l = sentence_token_features.size()

        if not return_local and not self.config.global_from_local:
            local_features = None
            local_mask = None
            N = None
        elif self.local_from_sentences:
            # (B x N_sent x d_l)
            sentence_features = \
                self.sentence_aggregator(sentence_token_features.view(B * N_sent, N_sent_tok, d_l),
                                         mask=sentence_token_mask.view(B * N_sent, N_sent_tok)).view(B, N_sent, d_l)
            sentence_features[sentence_mask.inverted_binary_mask] = 0.

            local_features = sentence_features
            local_mask = sentence_mask
            _, N, _ = sentence_features.size()
        else:
            raise NotImplementedError

        # compute global features by aggregating over all token features
        local_weights = None
        if return_global:
            if self.config.global_from_local:
                if self.config.local_weights_from_aggregator:
                    # (B x d_g)
                    global_features, local_weights = self.global_aggregator(local_features, mask=local_mask,
                                                                            return_weights=True)
                else:
                    # (B x N x d_l)
                    global_features = self.global_aggregator(local_features, mask=local_mask)
            else:
                # (B x d_l)
                global_features = self.global_aggregator(sentence_token_features.view(B, N_sent * N_sent_tok, d_l),
                                                         mask=sentence_token_mask.view(B, N_sent * N_sent_tok))
        else:
            global_features = None

        return EncoderOutput(local_features=local_features, global_features=global_features,
                             local_structure_size=N, local_mask=local_mask, local_weights=local_weights)

    def encode_joint(self, encoded, attention_mask, sentence_splits,
                     special_tokens_mask, return_global, return_local):

        # (B x N x d_language_model)
        token_features = encoded.last_hidden_state
        if self.input_projection:
            # (B x N x d_l)
            token_features = self.input_projection(token_features)
        # local from sentences
        # remove CLS and SEP token (assumes BERT-style encoding)
        token_features, attention_mask = remove_special_tokens(token_features, attention_mask, special_tokens_mask)
        if not return_local and not self.config.global_from_local:
            local_features = None
            local_mask = None
            N = None
        elif not self.local_from_sentences:
            local_features = token_features
            local_mask = attention_mask
            _, N, _ = token_features.size()
        else:
            # (B x N_sent x N_sent_tok x d_l)
            sentence_token_features, sentence_mask, sentence_token_mask = \
                restructure_reports_into_sentences(token_features, attention_mask, sentence_splits)
            sentence_mask = AttentionMask.from_binary_mask_or_attention_mask(sentence_mask,
                                                                             dtype=token_features.dtype)
            sentence_token_mask = AttentionMask.from_binary_mask_or_attention_mask(sentence_token_mask,
                                                                                   dtype=token_features.dtype)

            B, N, N_sent_tok, d_l = sentence_token_features.size()

            # (B x N_sent x d_l)
            sentence_features = \
                self.sentence_aggregator(sentence_token_features.view(B * N, N_sent_tok, d_l),
                                         mask=sentence_token_mask.view(B * N, N_sent_tok)).view(B, N, d_l)
            sentence_features[sentence_mask.inverted_binary_mask] = 0.

            local_features = sentence_features
            local_mask = sentence_mask
            _, N, _ = sentence_features.size()

        # compute global features by aggregating over all token features
        local_weights = None
        if return_global:
            if self.config.global_from_local:
                if self.config.local_weights_from_aggregator:
                    # (B x d_g)
                    global_features, local_weights = self.global_aggregator(local_features, mask=local_mask,
                                                                            return_weights=True)
                else:
                    # (B x N x d_l)
                    global_features = self.global_aggregator(local_features, mask=local_mask)
            else:
                # (B x N x d_l)
                global_features = self.global_aggregator(token_features, mask=attention_mask)
        else:
            global_features = None

        return EncoderOutput(local_features=local_features, global_features=global_features,
                             local_structure_size=N, local_mask=local_mask, local_weights=local_weights)

    def get_x(self, input_ids: torch.Tensor,
                sentence_splits: 'BatchSentenceSplitsInfo',
                attention_mask: Optional[Union['AttentionMask', torch.Tensor]],
                special_tokens_mask: torch.Tensor) -> 'ReportInputInfo':
        attention_mask = AttentionMask.from_binary_mask_or_attention_mask(attention_mask, dtype=self.language_model.dtype)

        # (B x N_sent x N_sent_tok)
        sentence_input_ids = input_ids.new_zeros((sentence_splits.batch_size,
                                                  sentence_splits.max_sentences_per_sample,
                                                  sentence_splits.max_tokens_per_sentence))
        if self.config.sentence_encoding_mode == 'joint':
            sentence_input_ids[sentence_splits.sentence_token_mask] \
                = input_ids[attention_mask.binary_mask * ~(special_tokens_mask.bool())]
        elif self.config.sentence_encoding_mode == 'independent':
            sentence_input_ids[sentence_splits.sentence_mask, :] = input_ids
        # list (batch dim) of list (sentence dim) of sentences
        sentences = [
            self.tokenizer.batch_decode(sample_sentence_input_ids, skip_special_tokens=True)
            for sample_sentence_input_ids in sentence_input_ids
        ]
        sentences = [
            [sent for sent in sample_sentences if len(sent) > 0] for sample_sentences in sentences
        ]

        return ReportInputInfo(sentences=sentences,
                               sentence_input_ids=sentence_input_ids,
                               sentence_splits=sentence_splits)

    def encode_text_queries(self, queries: List[str]) -> EncoderOutput:
        """

        :param queries: Queries consisting of one (independent) sentence each
        :return:
        """
        batch = [{'sentences': [sentence]} for sentence in queries]
        batch = self.batch_collator(batch)
        return self(**batch)

    @staticmethod
    def load(config_or_checkpoint: Union[ReportEncoderConfig, str], dataset_stats=None):
        if isinstance(config_or_checkpoint, str):
            return ReportEncoderModel.load_pretrained(config_or_checkpoint)
        else:
            return ReportEncoderModel(config_or_checkpoint)


@dataclass
class BatchSentenceSplitsInfo(TensorDataclassMixin):
    batch_size: int
    max_sentences_per_sample: int
    max_tokens_per_sentence: int
    sentence_token_mask: torch.Tensor  # (B x N_sent x N_sent_tok)
    sentence_mask: torch.Tensor  # (B x N_sent)
    token_same_sentence_mask: torch.Tensor  # (B x N_tok x N_tok)

    @staticmethod
    def compute_for_batch(sentences_start_positions_batch: List[List[int]], sentences_lengths_batch: List[List[int]]):
        B = len(sentences_start_positions_batch)
        max_sentences_per_sample = max(len(sample) for sample in sentences_lengths_batch)
        max_tokens_per_sentence = max(sentence_length for sample in sentences_lengths_batch for sentence_length in sample)

        sentence_mask = torch.zeros((B, max_sentences_per_sample), dtype=torch.bool)
        sentence_token_mask = torch.zeros((B, max_sentences_per_sample, max_tokens_per_sentence), dtype=torch.bool)
        max_tokens = max(sum(sample_sentence_lengths) for sample_sentence_lengths in sentences_lengths_batch)
        token_same_sentence_mask = torch.zeros((B, max_tokens, max_tokens))  # (B x N_tok x N_tok)

        for sample_index, (sentences_start_positions, sentences_lengths) \
                in enumerate(zip(sentences_start_positions_batch, sentences_lengths_batch)):
            num_sentences = len(sentences_start_positions)

            sentence_mask[sample_index, :num_sentences] = True
            for sentence_index, (sentences_start, sentences_length) \
                    in enumerate(zip(sentences_start_positions, sentences_lengths)):
                sentence_token_mask[sample_index, sentence_index, 0:sentences_length] = True
                token_same_sentence_mask[
                    sample_index,
                    sentences_start:sentences_start+sentences_length,
                    sentences_start:sentences_start+sentences_length
                ] = True

        return BatchSentenceSplitsInfo(batch_size=B,
                                       max_sentences_per_sample=max_sentences_per_sample,
                                       max_tokens_per_sentence=max_tokens_per_sentence,
                                       sentence_token_mask=sentence_token_mask,
                                       sentence_mask=sentence_mask,
                                       token_same_sentence_mask=token_same_sentence_mask)


@dataclass
class ReportInputInfo(TensorDataclassMixin):
    sentences: List[List[str]]
    sentence_input_ids: torch.Tensor  # (B x N_sent x N_sent_tok)
    sentence_splits: BatchSentenceSplitsInfo

    @staticmethod
    def is_report_input(dict_input: dict):
        return 'sentences' in dict_input

    @staticmethod
    def from_dict(dict_input):
        dict_input['sentence_splits'] = BatchSentenceSplitsInfo(**dict_input['sentence_splits'])
        return ReportInputInfo(**dict_input)

    def __getitem__(self, i):
        return ReportInputInfo(
            sentences=self.sentences[i],
            sentence_input_ids=self.sentence_input_ids[i],
            sentence_splits=self.sentence_splits[i],
        )


class ReportJointSentenceBatchCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        """
        Collates the list of outputs from ReportDataTransform into the batch that is given to forward.
        :param batch: list of samples (dicts) with:
            - sentences: List[str]
            Note: each element is an output from ReportDataTransform
        :return: dict with:
            - input_ids: torch.Tensor (B x N_tok)
            - attention_mask: torch.Tensor (B x N_tok)
            - special_tokens_mask: torch.Tensor (B x N_tok)
            - sentence_splits: BatchSentenceSplitsInfo

            This needs to match the (mandatory) inputs of forward
        """
        batch_sentences = [sample['sentences'] for sample in batch]
        # all the sentences in all samples of batch
        # len = number of all sentences in batch (sum of sentences of each sample of whole batch)
        all_sentences = [sent for sample in batch_sentences for sent in sample]
        # for each batch the related slice of all_sentences
        sentences_slices_of_samples = []
        start_index = 0
        for sample_sentences in batch_sentences:
            num_sentences = len(sample_sentences)
            sentences_slices_of_samples.append(slice(start_index, start_index + num_sentences))
            start_index += num_sentences
        max_content_length = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        # concatenate_sentences
        # 1. encode all sentences (without special tokens or padding)
        all_sentences = self.tokenizer.batch_encode_plus(
            all_sentences,
            max_length=max_content_length,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False
        )
        # 2. concatenate encoded sentences and add special tokens (and store sentence split positions)
        # for each sample, a list of all the start indices (token index) of its sentences
        sentences_start_positions_batch: List[List[int]] = []
        # for each sample, a list of all lengths (number of tokens) of its sentences
        sentences_lengths_batch: List[List[int]] = []
        # batch of all inputs ids
        input_ids_batch: List[List[int]] = []
        special_tokens_mask_batch: List[List[int]] = []
        for sentence_slice_of_sample in sentences_slices_of_samples:
            # list of the sentences (input_ids of each sentence) of this sample
            sentences_of_sample: List[List[int]] = all_sentences['input_ids'][sentence_slice_of_sample]

            input_ids_of_sample: List[int] = []
            current_start_index = 0
            sentences_start_positions: List[int] = []
            sentences_lengths: List[int] = []
            for sentence in sentences_of_sample:
                if current_start_index + len(sentence) <= max_content_length:
                    input_ids_of_sample.extend(sentence)
                    sentences_start_positions.append(current_start_index)
                    sentences_lengths.append(len(sentence))
                    current_start_index += len(sentence)
                elif current_start_index == 0:
                    # no sentence added
                    input_ids_of_sample.extend(sentence[:max_content_length])
                    sentences_start_positions.append(0)
                    sentences_lengths.append(max_content_length)
                    break
                else:
                    break
            input_ids_with_special_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids_of_sample)
            input_ids_batch.append(input_ids_with_special_tokens)
            special_tokens_mask_batch.append(self.tokenizer.get_special_tokens_mask(input_ids_with_special_tokens,
                                                                               already_has_special_tokens=True))
            sentences_start_positions_batch.append(sentences_start_positions)
            sentences_lengths_batch.append(sentences_lengths)
        # 3. pad and convert to PyTorch
        batch_sentences = self.tokenizer.pad(
            {'input_ids': input_ids_batch, 'special_tokens_mask': special_tokens_mask_batch},
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch_sentences['sentence_splits'] = BatchSentenceSplitsInfo.compute_for_batch(sentences_start_positions_batch,
                                                                                       sentences_lengths_batch)
        return batch_sentences


class ReportIndependentSentenceBatchCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        """
        Collates the list of outputs from ReportDataTransform into the batch that is given to forward.
        :param batch: list of samples (dicts) with:
            - sentences: List[str]
            Note: each element is an output from ReportDataTransform
        :return: dict with:
            - input_ids: torch.Tensor (N_batch_sent x N_sent_tok x d_language_model)
            - attention_mask: torch.Tensor (N_batch_sent x N_sent_tok x d_language_model)
            - special_tokens_mask: torch.Tensor (N_batch_sent x N_sent_tok x d_language_model)
            - sentence_splits: BatchSentenceSplitsInfo

            This needs to match the (mandatory) inputs of forward
        """
        batch_sentences = [sample['sentences'] for sample in batch]
        # all the sentences in all samples of batch
        # len = number of all sentences in batch (sum of sentences of each sample of whole batch)
        all_sentences = [sent for sample in batch_sentences for sent in sample]
        # (N_batch_sent x N_sent_tok x d_language_model)
        encoded_batch = self.tokenizer.batch_encode_plus(
            all_sentences,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )

        # for each sample, a list of all the start indices (token index) of its sentences
        sentences_start_positions_batch: List[List[int]] = []
        # for each sample, a list of all lengths (number of tokens) of its sentences
        sentences_lengths_batch: List[List[int]] = []

        start_index = 0
        sentence_lengths = encoded_batch['attention_mask'].sum(-1).int()  # (N_batch_sent)
        for sample in batch:
            num_sentences = len(sample['sentences'])

            sample_sentence_lengths = sentence_lengths[start_index:start_index + num_sentences]  # (num_sent)
            sentences_lengths_batch.append(sample_sentence_lengths.tolist())
            sample_start_indices = sample_sentence_lengths.cumsum(dim=0) - sample_sentence_lengths[0]
            sentences_start_positions_batch.append(sample_start_indices.tolist())

            start_index += num_sentences

        encoded_batch['sentence_splits'] = BatchSentenceSplitsInfo.compute_for_batch(sentences_start_positions_batch,
                                                                                       sentences_lengths_batch)
        return encoded_batch


def remove_special_tokens(token_features, attention_mask, special_tokens_mask):
    """

    :param token_features: (B x N x d)
    :param attention_mask: (B x N)
    :param special_tokens_mask: (B x N)
    :return:
    """
    normal_tokens_mask = ~(special_tokens_mask.bool())
    new_binary_mask = attention_mask.binary_mask * normal_tokens_mask

    token_features = token_features * normal_tokens_mask[:, :, None]

    return token_features, AttentionMask.from_binary_mask(new_binary_mask, dtype=token_features.dtype)


def restructure_reports_into_sentences(token_features: torch.Tensor, attention_mask,
                                       sentence_splits: BatchSentenceSplitsInfo):
    """

    :param token_features: (B x N_tok x d)
    :return:
        features: (B x N_sent x N_sent_tok x d)
        sentence_mask: (B x N_sent)
        token_mask: (B x N_sent x N_sent_tok)
    """
    B, N_tok, d = token_features.size()

    features = token_features.new_zeros((B,
                                         sentence_splits.max_sentences_per_sample,
                                         sentence_splits.max_tokens_per_sentence,
                                         d))
    features[sentence_splits.sentence_token_mask] = token_features[attention_mask.binary_mask]

    return features, sentence_splits.sentence_mask, sentence_splits.sentence_token_mask


def restructure_sentences_into_batch(all_token_features: torch.Tensor, sentence_splits: BatchSentenceSplitsInfo):
    """

    :param token_features: (B x N_tok x d)
    :return:
        features: (B x N_sent x N_sent_tok x d)
        sentence_mask: (B x N_sent)
        token_mask: (B x N_sent x N_sent_tok)
    """
    B_sentences, N_sent_tok, d = all_token_features.size()

    assert N_sent_tok == sentence_splits.max_tokens_per_sentence

    # (B x N_sent x N_sent_tok x d)
    features = all_token_features.new_zeros((sentence_splits.batch_size,
                                         sentence_splits.max_sentences_per_sample,
                                         N_sent_tok,
                                         d))
    features[sentence_splits.sentence_mask, :, :] = all_token_features

    return features, sentence_splits.sentence_mask, sentence_splits.sentence_token_mask
