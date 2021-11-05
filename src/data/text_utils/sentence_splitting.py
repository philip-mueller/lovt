from typing import Optional, List, Union, Tuple

import stanza
from nltk import PunktSentenceTokenizer, re
from nltk.tokenize.punkt import PunktParameters, PunktTrainer
from stanza import Document


PATTERN_REPLACE_MULTILINES = re.compile(r'(?:[\t ]*(?:\r\n|\n)+)+', flags=re.MULTILINE)
PATTERN_REPLACE_MULTISPACES = re.compile(r'[\t ]+')


def clean_text(text: str, replace_newlines=False, newline_token=' ') -> str:
    text = PATTERN_REPLACE_MULTILINES.sub('\n', text)

    if replace_newlines:
        text = text.replace('\n', newline_token)
    text = PATTERN_REPLACE_MULTISPACES.sub(' ', text).strip()
    return text


class SentenceSplitter:
    def __init__(self, lang: str = 'de',
                 train_text: str = None, abbreviations: list = None,
                 section_splitter=None, optional_sections: list = None):
        stanza.download(lang)
        self.stanza_tokenizer = stanza.Pipeline(lang, processors='tokenize', use_gpu=False)

        if train_text is not None:
            params = PunktTrainer(train_text).get_params()
        elif abbreviations:
            params = PunktParameters()
        else:
            params = None
        if abbreviations:
            for abbr in abbreviations:
                split_abbr = abbr.lower().split(' ')
                split_abbr = tuple(abbr_part[:-1] if abbr_part.endswith('.') else abbr_part for abbr_part in split_abbr)

                if len(split_abbr) == 1:
                    params.abbrev_types.add(split_abbr[0])
                elif len(split_abbr) == 2:
                    params.collocations.add((tuple(split_abbr)))
                else:
                    raise ValueError(abbr)

        self.punkt_tokenizer = PunktSentenceTokenizer(params) if params else None

        self.section_splitter = section_splitter
        self.optional_sections = optional_sections if optional_sections else []

    def __call__(self, report_full_text: str, **kwargs):
        if self.section_splitter is None:
            return report_full_text, self._process_section(report_full_text)
        else:
            sections = self.section_splitter(report_full_text, **kwargs)
            if sections is None:
                return None
            return {
                key: (report_full_text, *self._process_section(text))
                for key, text in sections.items()
            }

    def _process_section(self, section_txt: str) -> Tuple[List[str], int]:
        doc: Document = self.stanza_tokenizer(section_txt)

        sentences = [sent.text for sent in doc.sentences]
        if self.punkt_tokenizer:
            merged_sentences = []
            i = 0
            while i < len(sentences)-1:
                sent_a, sent_b = sentences[i], sentences[i+1]
                split_pos = len(sent_a) + 1
                merged_sentence = sent_a + ' ' + sent_b

                punkt_spans = self.punkt_tokenizer.span_tokenize(merged_sentence)
                punkt_split_positions = [span[0] for span in punkt_spans]

                if split_pos not in punkt_split_positions:
                    # punkt does not agree on split => ignore split
                    merged_sentences.append(merged_sentence)
                    i += 2
                else:
                    merged_sentences.append(sent_a)
                    i += 1
            sentences = merged_sentences

        sentences = [clean_text(sentence, replace_newlines=True) for sentence in sentences]

        return sentences, doc.num_tokens
