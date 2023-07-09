from typing import Any
import torch
import numpy as np
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMultipleChoice

class Longformer:
    def __init__(self, max_length = None) -> None:
        self.tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
        self.model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")
        self.model.eval()
        self.max_seq_length = 4096 if not max_length else max_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def get_max_seq_length(self):
        return self.max_seq_length
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_extra_input_length(self, question, options):
        return self.prepare_answering_input(question=question, options=options, context=" ")["input_ids"].shape[-1]

    def logits(self, context, question, options):
        with torch.no_grad():
            inputs = self.prepare_answering_input(question=question, options=options, context=context).to(self.device)
            outputs = self.model(**inputs)
        return outputs.logits

    def predict(self, context, question, options):
        with torch.no_grad():
            outputs = self.logits(question=question, options=options, context=context)
            prob = torch.softmax(outputs, dim=-1)[0].tolist()
        return np.argmax(prob)

    def prepare_answering_input(
        self,
        question,  # str
        options,   # List[str]
        context,   # str
        *args, **kwargs
    ):
        context = [context] * len(options)
        question_option = [question + " " + option for option in options]
        tokenized_examples = self.tokenizer(
            context, question_option,
            max_length=self.max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_examples['input_ids'] = tokenized_examples['input_ids'].unsqueeze(0)
        tokenized_examples['attention_mask'] = tokenized_examples['attention_mask'].unsqueeze(0)
        return tokenized_examples
    
class RobertaLarge:
    def __init__(self, max_length = None) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        self.model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        self.model.eval()
        self.max_seq_length = 512 if not max_length else max_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def get_max_seq_length(self):
        return self.max_seq_length
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_extra_input_length(self, question, options):
        return self.prepare_answering_input(question=question, options=options, context=" ")["input_ids"].shape[-1]
    
    def logits(self, context, question, options):
        inputs = self.prepare_answering_input(question=question, options=options, context=context).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)  
        return outputs.logits

    def predict(self, context, question, options):
        with torch.no_grad():
            outputs = self.logits(question=question, options=options, context=context)
            prob = torch.softmax(outputs, dim=-1)[0].tolist()
        return np.argmax(prob)
    
    def prepare_answering_input(
        self,
        question,  # str
        options,   # List[str]
        context,   # str
        *args, **kwargs
    ):  
        context = [context] * len(options)
        question_option = [question + " " + option for option in options]
        inputs = self.tokenizer(
            context,
            question_option,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors = 'pt'
        )    
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
        return inputs

class DebertaLarge:
    def __init__(self, max_length = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.model = AutoModelForMultipleChoice.from_pretrained("J4Q8/fine_tuned_deberta")
        self.model.eval()
        self.max_seq_length = 512 if not max_length else max_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def get_max_seq_length(self):
        return self.max_seq_length
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_extra_input_length(self, question, options):
        return self.prepare_answering_input(question=question, options=options, context=" ")["input_ids"].shape[-1]
    
    def logits(self, context, question, options):
        inputs = self.prepare_answering_input(question=question, options=options, context=context).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)  
        return outputs.logits

    def predict(self, context, question, options):
        with torch.no_grad():
            outputs = self.logits(question=question, options=options, context=context)
            prob = torch.softmax(outputs, dim=-1)[0].tolist()
        return np.argmax(prob)
    
    def prepare_answering_input(
        self,
        question,  # str
        options,   # List[str]
        context,   # str
        *args, **kwargs
    ):  
        context = [context] * len(options)
        question_option = [question + " " + option for option in options]
        inputs = self.tokenizer(
            context,
            question_option,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors = 'pt'
        )    
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
        return inputs