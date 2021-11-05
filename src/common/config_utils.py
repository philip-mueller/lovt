from deepdiff import DeepDiff
from omegaconf import OmegaConf


def prepare_config(config, config_cls, log):
    # make it possible to init this class with different types of configs (dataclass, omegaconf, dict)
    config = OmegaConf.create(config)
    # fill defaults, which is required if "deprecated" configs are used (e.g. when loading old checkpoints)
    config_defaults = OmegaConf.structured(config_cls)
    new_config = OmegaConf.merge(config_defaults, config)
    diff = DeepDiff(config, new_config, verbose_level=2)
    if len(diff) > 0:
        log.info(f'Defaults have been added to the config: {diff}')
    return new_config
