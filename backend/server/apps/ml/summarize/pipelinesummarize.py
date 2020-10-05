from transformers import pipeline
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class SummarizerPipeline():

    def __init__(self):
        tokenizer = T5Tokenizer.from_pretrained(
            '/Users/satyasiddharthdash/headlinegen/trained_model/best_tfmr/')
        model = T5ForConditionalGeneration.from_pretrained(
            '/Users/satyasiddharthdash/headlinegen/trained_model/best_tfmr')
        self.pipeline = pipeline(task="summarization",
                                 model=model, tokenizer=tokenizer)
        self.device = torch.device('cpu')

    def preprocess(self, input_data):

        input_data = dict(input_data)
        self.min_length = input_data['min_length']
        self.max_length = input_data['max_length']
        preprocessed_text = input_data['text'].strip().replace("\n", "")
        return preprocessed_text

    def predict(self, input_data):

        summary = self.pipeline(input_data,
                                max_length=self.max_length,
                                min_length=self.min_length)
        return summary[0]['summary_text']

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
