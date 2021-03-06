"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

from apps.ml.summarize.t5summarize import Summarizer
from apps.ml.summarize.pipelinesummarize import SummarizerPipeline
from apps.ml.registry import MLRegistry
import inspect
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()


try:
    registry = MLRegistry()
    rf = Summarizer()
    registry.add_algorithm(endpoint_name="summarize",
                           algorithm_object=rf,
                           algorithm_name="t5summarize",
                           algorithm_status="testing",
                           algorithm_version="0.0.1",
                           owner="Satya Siddharth Dash",
                           algorithm_description="Summarization algorithm with T5 with corresponding tokenizer",
                           algorithm_code=inspect.getsource(Summarizer))

    rf2 = SummarizerPipeline()
    registry.add_algorithm(endpoint_name="summarize",
                           algorithm_object=rf2,
                           algorithm_name="pipelinesummarize",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Satya Siddharth Dash",
                           algorithm_description="Summarization algorithm with finetuned transformer",
                           algorithm_code=inspect.getsource(SummarizerPipeline))


except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
