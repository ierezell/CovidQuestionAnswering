from datetime import datetime
from typing import List

import torch
from config import LANGUAGES
from .config import MODEL_NAMES
from datatypes import Answer, Chunk
from transformers import AutoModelForSeq2SeqLM as AutoModelSum
from transformers import AutoTokenizer
from transformers.pipelines import pipeline


class Summarizer:
    """
    Simlpe class which provide the ability to get
    the span answering the best the question given a context paragraph

    It is instanciated in :func:`~indexer.indexer.create_models`
    which is called by :func:`~indexer.indexer.process`

    It is used in :func:`~qa.refinder.answer_question`
    """

    def __init__(self, lang: LANGUAGES = 'multi', prefer_gpu: bool = False
                 ) -> None:

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[lang])
        model = AutoModelSum.from_pretrained(MODEL_NAMES[lang])

        DEVICE = 0 if (torch.cuda.is_available() and prefer_gpu) else -1

        self.summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            framework='pt',
            device=DEVICE
        )

        print(f"Summary will be computed on {'cpu' if DEVICE < 0 else 'gpu'}")

    def summarize(self, supports: List[Chunk]) -> Answer:
        all_text = " ".join([chunk['content'] for chunk in supports[:5]])
        if not all_text:
            answer: Answer = {
                'score': 0.0,
                'content': "",
                'answer': "",
                'title': "",
                'date': datetime.now(),
                'start': 0,
                'end': 1,
                'elected': 'n/a',
                'link': ['']
            }
            return answer
        with torch.no_grad():
            summary = self.summarizer(all_text, min_length=50, max_length=500)
            answer: Answer = {
                'score': 1.0,
                'content': all_text,
                'answer': summary,
                'title': " / ".join([chunk['title'] for chunk in supports[:5]]),
                'date': datetime.now(),
                'start': 0,
                'end': len(summary),
                'elected': 'n/a',
                'link': ['']
            }

        return answer
