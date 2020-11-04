from typing import Dict
from config import LANGUAGES

MODEL_NAMES: Dict[LANGUAGES, str] = {
        "fr": "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",  # noqa E501
        "multi": 'facebook/mbart-large-cc25',
        "en": 'None',
        "qa_en": 'None',
        "qa_fr": 'etalab-ia/camembert-base-squadFR-fquad-piaf',
        "qa_multi": 'deepset/xlm-roberta-large-squad2',
        'sum_fr': "facebook/bart-large-xsum"}
