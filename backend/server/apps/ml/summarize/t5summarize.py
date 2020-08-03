import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class Summarizer:

    def __init__(self):

        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.device = torch.device('cpu')

    def preprocess(self, input_data):

        input_data = dict(input_data)
        self.min_length = input_data['min_length']
        self.max_length = input_data['max_length']
        preprocessed_text = "summarize: " + \
            input_data['text'].strip().replace("\n", "")
        return preprocessed_text

    def predict(self, input_data):

        tokenized_text = self.tokenizer.encode(input_data,
                                               return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(tokenized_text,
                                          min_length=self.min_length,
                                          max_length=self.max_length)
        output = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)

        return output

    def postprocess(self, input_data):
        if self.min_length <= len(input_data.strip().split(' ')) <= self.max_length:
            return {'summary': input_data, 'length': len(input_data.strip().split(' ')), "status": "OK"}

    def compute_prediction(self, input_data):

        try:
            input_data = self.preprocess(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocess(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
