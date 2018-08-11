## Transfer Learning for Abstractive Text Summarization with Pointer-Generator Networks

Based on https://arxiv.org/abs/1704.04368

### Downloading the datasets

The CNN/Daily Mail dataset can be obtained from [here](https://cs.nyu.edu/~kcho/DMQA/).

The NYT Annotated corpus can be ontained from [here](https://catalog.ldc.upenn.edu/ldc2008t19)

### Generating data for training

To generate the training data from CNN/Daily Mail datasets, run the following command. The downloaded dataset is expected to be in the `data/` folder. This will write the tokeninzed data to `data/tokens` folder and serialized tensorflow examples to `data/bins`.

` python create_datafiles.py cnn-dailymail`

To generate the NYT data, run the following command.

` python create_datafiles.py nyt`


Most of the sequence to sequence attention model code is borrowed from [tensorflow](https://github.com/tensorflow/models/tree/master/research/textsum)


### Experiments conducted

1. Trained a pointer-gen on NYT corpus. The trained model is available [here]()

2. Evaluated a pre-trained model on NYT corpus which is available [here](https://drive.google.com/file/d/0B7pQmm-OfDv7ZUhHZm9ZWEZidDg/view)

3. Fine tuned the pre-trained model with few example from NYT corpus and evaluated the model. This model is available [here]()


### Results
11,000 examples from NYT corpus were sampled to generate sumamries and were evaluated using the ROUGE-2.0 java library. The scores in 95% confidence interval below:

1. Baseline scores:
ROUGE-1 Average_F: (0.3102, 0.3357)
ROUGE-2 Average_F: (0.1891, 0.2120)
ROUGE-L Average_F: (0.2376, 0.2602)

2. Pre-trained model scores:
ROUGE-1 Average_F: (0.1762, 0.1949)
ROUGE-2 Average_F: (0.0855, 0.1004)
ROUGE-L Average_F: (0.1716, 0.1903)

3. Scores for the fine-tuned model
ROUGE-1 Average_F: (0.2965, 0.3229)
ROUGE-2 Average_F: (0.1792, 0.2027)
ROUGE-L Average_F: (0.2351, 0.2586)
