# Recurrent Neural Networks - Advanced Machine Learning 2021-II

We have seen in class that the use of Recurrent Neural Networks is wide and common in applications requiring the modelling of sequential data. This homework is divided into two stages, each one exploring one of these fundamental RNN applications. In particular, the first stage targets **text generation**. You will be training with a small dataset of known jokes in English, with the objective of generating new jokes given some initial words. This stage will allow you to explore the concept of representation, embeddings and the basic PyTorch implementation of LSTMs. During the second stage of this homework, you will explore the task of **image captioning**. To do so, you will make use of a CNN encoder-RNN decoder architecture with a subset of the renowned COCO dataset. 

## Objectives

- To understand the implementation of RNN architectures to solve common tasks in NLP and its intersection with CV.
- To recognize the meaning and structure of data representation and embedding, when serving as input to RNNs.
- To comprehend LSTM and GRU theoretical and practical differences.
- To explore the synergy between CNN and RNN to adress tasks at the intersection of NLP and CV.

## Main requirements

- nltk = 3.5
- pandas = 1.3.2
- pycocotools = 2.0.2
- python = 3.9.6
- torchvision = 0.10.0
	
## Homework

### Part 1: Text generation

After training the base model, you should be able to predict *n* consecutive words to a given sample. For instance, in response to an input of "Knock knock. Who's there?", the model could predict the following 100 words:

['Knock', 'knock.', 'Whos', 'there?', 'number', 'jumper', 'jumper', 'cables?', 'You', 'better', 'not', 'try', 'to', 'start', 'anything.', "Don't", 'you', 'hate', 'jokes', 'about', 'German', 'sausage?', "They're", 'the', 'wurst!', 'Two', 'artists', 'had', 'an', 'art', 'contest...', 'It', 'ended', 	'in', 'a', 'draw', 'Why', 'did', 'the', 'paper', 'cross', 'the', 'road?', 'To', 'get', 'to', 'the', "moron's", 'house.', '*knock', 'knock*', '^^Whose', '^^there?', '*the', 'chicken...*', 'Why', 'did', 'the', 'SSD', 'burn', 'an', 'monk', 'B', 'Now', 'if', 'your', 'beet.', 'Why', 'are', 'music', 'not', 'travelers', 'Man', 'Man', '(as', 'on', 'the', 'desert', 'and', 'use', 'home?', 'and', 'hard.', '*Al', 'dente*,', 'you', 'tell', 'the', 'hair', 'P.', 'Well,', 'two', 'hold', 'only', 'a', 'long', 'spill.', 'from', 'me', 'go', 'I', 'went', 'to', 'a']

**Task # 1** (0.75 pts)

a. Explore the *dataset.py*, *train.py* and *predict.py* files to understand how the representation and embedding of words and phrases is done (Hint: Check how the information is entering the network and the functions used to encode input data). In your report, describe and compare at least two different techniques of data representation and embedding used in Natural Language Processing. 

b. Discuss how this implementation should be modified to be used in practice for **text classification**. In a specific manner, describe the changes you would implement in the input data encoding, architecture and overall information flow. 

**Task # 2**

Divide the data into train and validation splits to perform the following experimentation:

a. Experiment with the model capacity by adding more LSTM layers (0.75 pts).

b. Modify the LSTM hyperparameters in order to obtain better results (qualitatively, better jokes). You can experiment with parameters of the input representation and/or LSTM architecture (1 pt). 

In your report, comment and discuss your results. Be aware of your evaluation and comparison strategy between models. Think about the fundamental difficulties in quantifying performance in the text generation task. Gather your ideas for **bonus question B)** and **Task # 4** in the following section. 

**Bonus questions** (0.5 pts)

a. In its native implementation (given hyperparameters and training curriculum), is the model overfitting to the data? If so, comment on the decisions taken by the authors that might have lead to this situation.

b. How are you measuring performance? Should one concentrate on measuring similarity to phrases in the validation set? Is there another indirect measure of learning in this specific task?


### Part 2: Image captioning

The dataset used for this section of the homework is a sample of the MS COCO 2017 dataset and its corresponding annotations. In particular, you will be using the 5,000 validation images, which were reorganized into train and test splits. 

  	annotations/   # annotation json files
  	imgs/train/    # Training images (3,500)
  	imgs/test/     # Test images (1,500)
	
<p align="center"><img src="https://cocodataset.org/images/coco-logo.png" width="400"/></p>

Before you start, make sure to have access to the data. To do so, make a softlink to the COCO dataset inside the *image-captioning* directory. Note: this access should be at the same level of the .py codes. You can do this on both BCV002 and BCV003 by:

		BCV002: ln -s "/media/user_home0/fescallon/COCO_data/"
		BCV003: ln -s "/media/SSD0/datasets/COCO_data/"

  
First things first: explore and understand the *data_loader.py*, *model.py*, *train.py* and *inference.py* documents.

**Task # 3** (1.75 pts)

Experiment with two different pretrained models for the Encoder. In addition, implement a GRU instead of the native LSTM. In this sense, you should have at least four experiments by combining these variations. Compare, contrast and discuss the results. 

You will need to think about a fair manner to compare performances between architectures (qualitatively). 

**Task # 4** (0.75 pts)

In your report, answer the following question: What metric or evaluation system do you think could be useful for this task? We all know different language words could mean similar things. Is this a problem? If so, investigate or propose an alternative for the evaluation of this task.


**Note:** As commented during class, you should be able to obtain satisfactory results, but could also obtain some results far from the images' contents. Take a look a these examples:

<img src="https://user-images.githubusercontent.com/69943932/131237461-dafc72e5-3596-4147-8526-a25617de1cb6.png" width="430"> <img src="https://user-images.githubusercontent.com/69943932/131237467-9b63fa62-c9c9-49e0-ba50-315a2adbc4d5.png" width="400">

## Additional resources

Inside the directory *Papers*, you will find a small "arXiv" that gathers potentially useful RNN resources. 

## Deadline

September 13, 2021 - 11:59 p.m.

## Sources

This repository is an adaptation of two external implementations.

Text generation:

https://github.com/closeheat/pytorch-lstm-text-generation-tutorial

https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial

Image captioning:

https://github.com/tatwan/image-captioning-pytorch
