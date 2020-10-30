from transformers import AutoTokenizer, AutoModel
from typing import List
from datatypes import MODEL_NAMES, LANGUAGES, EMBED_DIM
import torch

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Embeddings will be computed on {DEVICE}")


class Embedder:
    def __init__(self, processor, lang: LANGUAGES = 'multi') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[lang])
        self.model = AutoModel.from_pretrained(MODEL_NAMES[lang])
        self.model = self.model.to(DEVICE).eval()
        self.processor = processor

    def embed(self, text: str) -> List[float]:
        with torch.no_grad():
            # embeded = self.model(**self.tokenizer(text, return_tensors='pt'))[0][0]
            # return embeded[0].tolist()

            sentences = [sent.text for sent in self.processor(text).sents]
            if not sentences:
                sentences = [text]

            tokens = self.tokenizer(sentences, return_tensors='pt',
                                    padding=True, add_special_tokens=True)

            tokens = tokens.to(DEVICE)

            if tokens['input_ids'].size()[1] > 512:
                print(text)
                raise RuntimeError(
                    "There is too much tokens in the text above")
            sentences_embeded = self.model(**tokens)[0][:, 0, :]

            embeded = torch.mean(sentences_embeded, dim=0).cpu().tolist()
        return embeded
