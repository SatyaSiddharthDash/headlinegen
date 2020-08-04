from django.test import TestCase
import inspect

from apps.ml.summarize.t5summarize import Summarizer
from apps.ml.summarize.pipelinesummarize import SummarizerPipeline
from apps.ml.registry import MLRegistry


class MLTests(TestCase):
    def test_summarization(self):
        input_text = """
                    The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world. At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors."We'll be the comeback kids, all of us," he said. "We want to get our country back." The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
                    """
        input_data = {'text': input_text, 'min_length': 5, 'max_length': 100}
        summarizer = Summarizer()
        response = summarizer.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue(
            5 <= len(response['summary'].strip().split(' ')) <= 100)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "summarize"
        algorithm_object = Summarizer()
        algorithm_name = "t5summarize"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Satya Siddharth Dash"
        algorithm_description = "Summarization algorithm with T5 with corresponding tokenizer"
        algorithm_code = inspect.getsource(Summarizer)
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)

    def test_pipelinesummarization(self):
        input_text = """
                    The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world. At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors."We'll be the comeback kids, all of us," he said. "We want to get our country back." The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
                    """
        input_data = {'text': input_text, 'min_length': 5, 'max_length': 100}
        summarizer = SummarizerPipeline()
        response = summarizer.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue(
            5 <= len(response['summary'].strip().split(' ')) <= 100)
