from dataclasses import dataclass
from typing import Optional, Tuple, Collection

import torch
from torch import distributions


@dataclass
class ReportAugmentationConfig:
    augment: bool = True

    # None: original order, 'random_swaps': apply random sentence swaps, ('random_permute': apply random permutation)
    sentence_shuffling: Optional[str] = None
    swap_prob: float = 0.6  # geometric dist with p=1-swap_prob

    # None: take all sentences, 'random_sentence': sample 1 sentence, 'random_crop': sample subset of sentences (1 to all)
    sentence_sampling: Optional[str] = 'random_sentence'
    sentence_crop_range: Collection[float] = (0.6, 1.0)

    duplicate_sentences: bool = False
    duplication_prob: float = 0.2  # geometric dist with p=1-duplication_prob


class ReportDataTransform:
    def __init__(self, config: ReportAugmentationConfig, val=False):
        assert config.sentence_shuffling in (None, 'random_swaps')

        augment = False if val else config.augment

        if not augment or config.sentence_shuffling is None:
            self.sentence_shuffler = None
        elif config.sentence_shuffling == 'random_swaps':
            self.sentence_shuffler = RandomIndexSwapper(config.swap_prob)
        else:
            raise ValueError(config.sentence_shuffling)

        assert config.sentence_sampling in (None, 'random_crop', 'random_sentence')
        if not augment or config.sentence_sampling is None:
            self.sentence_sampler = None
        elif config.sentence_sampling == 'random_crop':
            self.sentence_sampler = IndexSubregionSampler(config.sentence_crop_range)
        elif config.sentence_sampling == 'random_sentence':
            self.sentence_sampler = sample_single_index

        if augment and config.duplicate_sentences:
            self.sentence_duplicator = IndexDuplicator(config.duplication_prob)
        else:
            self.sentence_duplicator = None

    def __call__(self, report_sample):
        """

        Options on sentence level:
        - change sentence order (1. sample number of flips (gamma), 2. for each flip sample length (gamma) and first sentence (uniform)
        - subset of sentences (1. sample length (???), 2. sample start sentence(uniform)) / delete sentences
        - repeat sentences

        Options on word level:
        - randomly remove/replace/add words

        :param report_sample: Dict with
            - sentences: List[str]
        :return: Dict with
            "sentences": List[str]
            "num_sentences": int
        """
        sentences = report_sample['sentences']

        if self.sentence_shuffler or self.sentence_sampler or self.sentence_duplicator:
            indices = torch.arange(end=len(sentences), dtype=torch.int)
            if self.sentence_shuffler:
                indices = self.sentence_shuffler(indices)
            if self.sentence_sampler:
                indices = self.sentence_sampler(indices)
            if self.sentence_duplicator:
                indices = self.sentence_duplicator(indices)
            assert len(indices) > 0
            sentences = [sentences[i] for i in indices]

        return {'sentences': sentences}


def sample_single_index(indices: torch.Tensor) -> torch.Tensor:
    return indices[torch.randint(high=len(indices), size=(1,))]


class RandomIndexSwapper:
    def __init__(self, swap_prob: float = 0.5):
        self.swap_dist = distributions.Geometric(probs=1-swap_prob)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        num_swaps = int(self.swap_dist.sample().item())
        swap_indices = torch.randint(high=len(indices), size=(num_swaps, 2))

        for from_index, to_index in swap_indices:
            indices[[from_index.item(), to_index.item()]] = indices[[to_index.item(), from_index.item()]]

        return indices


class IndexSubregionSampler:
    def __init__(self, region_size_range: Tuple[float, float] = (0.2, 1.0)):
        self.region_size_range = region_size_range

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        min_length = max(1, round(self.region_size_range[0] * len(indices)))
        max_length = max(1, round(self.region_size_range[1] * len(indices)))

        length = torch.randint(low=min_length, high=max_length+1, size=())
        start_index = torch.randint(high=len(indices)-(length-1), size=())
        return indices[start_index:start_index+length]


class IndexDuplicator:
    def __init__(self, duplication_prob: float = 0.3):
        self.num_duplication_dist = distributions.Geometric(probs=1 - duplication_prob)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        num_duplications = int(self.num_duplication_dist.sample().item())
        if num_duplications > 0:
            # (num_duplications)
            duplicated_indices = torch.multinomial(torch.ones(len(indices)),
                                                   num_duplications,
                                                   replacement=True)
            duplicate_insertion_indices = torch.multinomial(torch.ones(len(indices)+num_duplications),
                                                            num_duplications,
                                                            replacement=False)
            original_insertion_mask = torch.ones(len(indices) + num_duplications, dtype=torch.bool)
            original_insertion_mask[duplicate_insertion_indices] = False

            new_indices = torch.empty(len(indices) + num_duplications, dtype=torch.int)
            new_indices[original_insertion_mask] = indices
            new_indices[duplicate_insertion_indices] = indices[duplicated_indices]
        else:
            new_indices = indices

        return new_indices
