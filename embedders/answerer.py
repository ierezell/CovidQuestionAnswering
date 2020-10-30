from typing import List

from datatypes import LANGUAGES, MODEL_NAMES
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.pipelines import pipeline

import torch


class Answerer:
    def __init__(self, lang: LANGUAGES = 'multi') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[lang])
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAMES[lang])
        self.nlp = pipeline("question-answering",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            framework='pt',
                            device=0 if torch.cuda.is_available() else -1)

    def answer(self, question: str, context: str) -> List[float]:
        with torch.no_grad():
            res = self.nlp(question=question, context=context)
        return res
        # OR
        # inputs = self.tokenizer(question, context, add_special_tokens=True,
        #                         return_tensors="pt")
        # input_ids = inputs["input_ids"].tolist()[0]

        # text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # answer_start_scores, answer_end_scores = self.model(**inputs)

        # answer_start = torch.argmax(answer_start_scores)
        # answer_end = torch.argmax(answer_end_scores) + 1

        # answer = self.tokenizer.convert_tokens_to_string(
        #     self.tokenizer.convert_ids_to_tokens(
        #         input_ids[answer_start:answer_end]
        #     )
        # )

        # score = torch.max(answer_end_scores)+torch.max(answer_end_scores) / 2
        # return {'score': score.item(),
        #         'start': answer_start,
        #         'end': answer_end,
        #         'answer': answer}
