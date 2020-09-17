# headlinegen
A scalable tool to perform abstractive text-summarization to generate headlines.
This project was done as a part of the Springboard AI/ML track.
The following is the project proposal as submitted to Springboard. For a few technical details click [here](https://github.com/SatyaSiddharthDash/headlinegen/techinical-details.md)

## Project Proposal
**Motivation**: Text-summarization is one of the most important applications of Natural Language Processing. With recent advances in NLP due to the leaps and bounds in deep learning, Automatic Text Summarization has also seen great improvements. From simple keyword and n-gram summarization, followed by simple sequence-to-sequence models, all the way to deep transformers and encoder-decoder networks, text summarization has come a long way. While improvements are usually reported as achieving State-of-the-art results on metrics such as ROGUE-I and ROGUE-II, text summarization is very application-specific. Transfer learning has helped immensely in this regard. Deep pre-trained networks can now be fine-tuned to specific tasks such as summarization. Methods such as parameter-sharing and knowledge distillation have led to extremely light models that tradeoff very little accuracy for a huge improvement in computational constraints. I intend to explore the various text-summarization methods in use today and also create a scalable model myself.

**Extractive vs Abstractive Summarization**: Text-summarization can be clearly distinguished into two types. Extractive summarization involves extracting the most important matter from a given text. This is a supervised learning task as the summaries(or summary sentences) are within the data. Extractive text summarization involves weighing the importance of the various sentences within the text and giving the most important ones as the output. Abstractive summarization, on the other hand, is more difficult and is harder to evaluate for performance. It is an unsupervised technique which involves Natural Language Understanding, followed by Natural Language Generation. In this project, I will dive into the techniques and methods used in abstractive summarization.

**Abstractive summarization--a brief overview**: As outlined in [1] older methods of abstractive summarization can be broadly classified into structure-based and semantic-based methods. Each of these methods includes a wide variety of methods that have been used. While research in extractive summarization reached maturity a while ago, abstractive summarization has achieved major breakthroughs in the past few years due to deep learning. Models such as BERT, BART and most recently T5 [2], have achieved state-of-the-art results in various tasks including abstractive summarization.

**Objective**: This project will explore the various methods used to perform abstractive text-summarization and outline the advancements, over the last few years. It will attempt to lay bare the various considerations that were taken into account while developing these methods. Then, I will create a text-summarization tool using the T5 model from the transformers library from Huggingface and scale it to train quickly using cloud resources. The transformers library contains both the tokenizer as well as the model for the same. The model will be evaluated using ROUGE-I, ROUGE-II and ROUGE-L scores. I will be using the NYTimes dataset for training. The reasons for choosing the above-mentioned dataset are:- 
The dataset size is good enough for very deep models.
The dataset is more diverse than other datasets for the same task(such as GIGAWORD and CNN-Daily Mail).
The dataset has been experimented with using traditional models.
The data is relatively clean and needs very less preprocessing. 


**Technical considerations and computational requirements**: The text summarization tool is to be built using the PyTorch library and initially trained on a small subset of the data on the cloud on the PaperspaceⓇ platform to try and see if things are working out. I will be using a machine with an 8-core CPU and a single Nvidia P5000 GPU. I will serve the models as a Django API. The entire production code will be scalable using docker containers.

**Final outcome**: I aim to deliver a working scalable model that can perform abstractive summarization.

References:
Som Gupta, S. K Gupta, Abstractive summarization: An overview of the state of the art, Expert Systems with Applications, Volume 121, 2019, Pages 49-65, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2018.12.011. (http://www.sciencedirect.com/science/article/pii/S0957417418307735)

Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." arXiv preprint arXiv:1910.10683 (2019).
