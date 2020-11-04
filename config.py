from typing import Literal, Dict


LANGUAGES = Literal['en', 'fr', 'multi',
                    'qa_fr', 'qa_en', 'qa_multi', 'sum_fr']
CODE_TO_LANG = {"en": "english", "fr": "french"}
LANG_TO_CODE = {v: k for k, v in CODE_TO_LANG.items()}

CHUNK_SIZE = 1000  # number of word per paragraph
NB_KEYWORDS = 6  # number of keywords to keep per chunk
EMBED_DIM = 768


SPACY_MODEL_NAMES: Dict[LANGUAGES, str] = {"multi": "xx_ent_wiki_sm",
                                           "en": "en_core_web_sm",
                                           "fr": "fr_core_news_sm"}
