## Transfer Learning for Abstractive Text Summarization with Pointer-Generator Networks

### Downloading the datasets

The CNN/Daily Mail dataset can be obtained from [here](https://cs.nyu.edu/~kcho/DMQA/).

The NYT Annotated corpus can be ontained from [here](https://catalog.ldc.upenn.edu/ldc2008t19)

### Generating data for training

To generate the training data from CNN/Daily Mail datasets, run the following command. The downloaded dataset is expected to be in the `data/` folder. This will write the tokeninzed data to `data/tokens` folder and serialized tensorflow examples to `data/bins`.

` python create_datafiles.py cnn-dailymail`

To generate the NYT data, run the following command.

` python create_datafiles.py nyt`
