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
        return max([self.prepare_answering_input(question=question, options=option, context=" ")["input_ids"].shape[-1] for option in options])

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
        return max([self.prepare_answering_input(question=question, options=option, context=" ")["input_ids"].shape[-1] for option in options])
    
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
            padding="longest",
            truncation=True,
            return_tensors = 'pt'
        )    
        input_ids = inputs['input_ids'].unsqueeze(0)
        attention_mask = inputs['attention_mask'].unsqueeze(0)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return encoded


# #Deberta 
#     from transformers import AutoModel, AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer
#     model_name = "sileod/deberta-v3-base-tasksource-nli"
#     task_name = "cosmos_qa"
#     task = tasksource.load_task(task_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name,ignore_mismatched_sizes=True)
#     adapter = Adapter.from_pretrained(model_name.replace('-nli','')+'-adapters')
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = adapter.adapt_model_to_task(model, task_name)
#     model.config.id2label=str(task['train'].features['labels'])

#     task_index = adapter.config.tasks.index(task_name)

#     with torch.no_grad():
#         model.deberta.embeddings.word_embeddings.weight[tokenizer.cls_token_id]+=adapter.Z[task_index]

#     #can do model inference now yay!

#     pipe = TextClassificationPipeline(
#     model=model, tokenizer=tokenizer)