from django.test import TestCase
from rest_framework.test import APIClient


class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_text = """
                    The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world. At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors."We'll be the comeback kids, all of us," he said. "We want to get our country back." The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
                    """
        input_data = {'text': input_text, 'min_length': 5, 'max_length': 100}
        classifier_url = "/api/v1/summarize/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue("summary" in response.data)
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)
