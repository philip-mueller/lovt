from enum import Enum
from typing import Tuple

from transformers import PretrainedConfig, BertForMaskedLM, BertModel, \
    PreTrainedModel, PreTrainedTokenizerBase, BertTokenizerFast


class ModelTask(Enum):
    MASKED_LM = 1


MODEL_MAPPINGS = {
    'bert': {
        None: BertModel,
        ModelTask.MASKED_LM: BertForMaskedLM
    }
}

TOKENIZER_MAPPINGS = {
    'bert': BertTokenizerFast,
}


def load_language_model(pretrained_model: str, pretrained_tokenizer: str = None, model_task: ModelTask = None,
                        do_load_tokenizer=True,
                        do_load_model=True) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if pretrained_tokenizer is None:
        pretrained_tokenizer = pretrained_model
    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model)
    model_type = config_dict["model_type"]

    if do_load_model:
        model_class = MODEL_MAPPINGS[model_type][model_task]
        model = model_class.from_pretrained(pretrained_model)
    else:
        model = None

    if do_load_tokenizer:
        tokenizer = TOKENIZER_MAPPINGS[model_type].from_pretrained(pretrained_tokenizer)
    else:
        tokenizer = None
    return model, tokenizer
