from typing import Any
import torch
import numpy as np
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from transformers import AutoModelWithLMHead, AutoTokenizer

class Longformer:
    def __init__(self) -> None:
        self.tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
        self.model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")
        self.max_seq_length=4096
    
    def get_max_seq_length(self):
        return self.max_seq_length
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_extra_input_length(self, question, options):
        return self.prepare_answering_input(question=question, options=options, context=" ")["input_ids"].shape[-1]

    def predict(self, context, question, options):
        inputs = self.prepare_answering_input(tokenizer=self.tokenizer, question=question, options=options, context=context)
        outputs = self.model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        return np.argmax(prob)

    def prepare_answering_input(
        self,
        question,  # str
        options,   # List[str]
        context,   # str
    ):
        c_plus_q   = context + ' ' + self.tokenizer.bos_token + ' ' + question 
        c_plus_q_4 = [c_plus_q] * len(options)
        tokenized_examples = self.tokenizer(
            c_plus_q_4, options,
            max_length=self.max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized_examples['input_ids'].unsqueeze(0)
        attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
        example_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return example_encoded
    
class RobertaLarge:
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        self.model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        self.max_seq_length=512

    def get_max_seq_length(self):
        return self.max_seq_length
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_extra_input_length(self, question, options):
        return self.prepare_answering_input(question=question, options=options, context=" ")["input_ids"].shape[-1]
    
    def predict(self, context, question, options):
        inputs = self.prepare_answering_input(question=question, options=options, context=context)
        outputs = self.model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        return np.argmax(prob)
    
    def prepare_answering_input(
        self,
        question,  # str
        options,   # List[str]
        context,   # str
    ):  
        context = [context] * len(options)
        question_option = [question + " " + option for option in options]
        inputs = self.tokenizer(
            context,
            question_option,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=False,
        )    

        input_ids = inputs['input_ids'].unsqueeze(0)
        attention_mask = inputs['attention_mask'].unsqueeze(0)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return encoded
