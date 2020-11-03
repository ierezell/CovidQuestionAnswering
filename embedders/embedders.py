from typing import List, Literal

import torch
from transformers import AutoModel, AutoTokenizer

from config import LANGUAGES

from .config import MODEL_NAMES


class Embedder:
    """
    A class to help the embedding of any text

    It is instanciated in :func:`~indexer.indexer.create_models`
    which is called by :func:`~indexer.indexer.process`

    It is used in :func:`~indexer.metabuilder.embed_chunk`
    It is used in :func:`~qa.retriever.retrieve_docs`
    """

    def __init__(self, processor, lang: LANGUAGES = 'multi',
                 prefer_gpu: bool = False) -> None:

        self.device = torch.device("cuda") if (
            torch.cuda.is_available() and prefer_gpu) else torch.device("cpu")

        print(f"Embedder will compute on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[lang])
        self.model = AutoModel.from_pretrained(MODEL_NAMES[lang])
        self.model = self.model.to(self.device).eval()
        self.processor = processor

    def embed(self, text: str, method: Literal['all', 'sentence']
              ) -> List[float]:
        """Embed any text to a vector

        Args:
            text (str)
            method (all or sentence): embed all the paragraph
                            or compute the mean of all sentence embeddings

        Raises:
            RuntimeError: If the tokenized text gives too much tokens then the
            embedder will crash. Catching and preventing the error before.

            AssertionError: Check that the method is all or sentence.
        """
        with torch.no_grad():

            if method == 'all':
                # Compute all the paragraph at once
                tokens = self.tokenizer(text, return_tensors='pt',
                                        add_special_tokens=True)
                tokens = tokens.to(self.device)

                embeded = self.model(**tokens)[0][0]

                return embeded[0].tolist()

            elif method == 'sentence':
                # Compute embedding for each sentence and return the mean
                sentences = [sent.text for sent in self.processor(text).sents]
                if not sentences:
                    sentences = [text]

                tokens = self.tokenizer(sentences, return_tensors='pt',
                                        padding=True, add_special_tokens=True)

                tokens = tokens.to(self.device)

                if tokens['input_ids'].size()[1] > 512:
                    raise RuntimeError(
                        f"There is more than 512 tokens in the text : {text}"
                    )

                # Take all the CLS tokens for each sentence
                sentences_embeded = self.model(**tokens)[0][:, 0, :]

                embeded = torch.mean(sentences_embeded, dim=0).cpu().tolist()
                return embeded
            else:
                raise AssertionError(
                    "[method] parameter should be all or sentence"
                )
