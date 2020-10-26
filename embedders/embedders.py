from transformers import AutoTokenizer, AutoModel
from typing import List
from datatypes import MODEL_NAMES, LANGUAGES


class Embedder:
    def __init__(self, lang: LANGUAGES = 'multi') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[lang])
        self.model = AutoModel.from_pretrained(MODEL_NAMES[lang])

    def embed(self, text: str) -> List[float]:
        embeded = self.model(**self.tokenizer(text, return_tensors='pt'))[0][0]
        return embeded[0].tolist()

    def tokenize(self, text: str):
        return self.tokenizer(text)['input_ids']
