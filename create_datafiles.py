import os, sys
import pathlib
import subprocess
import tensorflow as tf
import hashlib

from helpers import constants, utils, vocabulary

TRAIN_URLS_FILES = "data_splits/all_train.txt"
VAL_URLS_FILES = "data_splits/all_val.txt"
TEST_URLS_FILES = "data_splits/all_test.txt"

TOKENIZED_TRAIN_FILES = "data/tokens/train"
TOKENIZED_VAL_FILES = "data/tokens/val"
TOKENIZED_TEST_FILES = "data/tokens/test"
TOKENIZED_SAMPLE_FILES = "data/tokens/samples"

TF_EXAMPLE_TRAIN_FILES = "data/bins/train"
TF_EXAMPLE_VAL_FILES = "data/bins/val"
TF_EXAMPLE_TEST_FILES = "data/bins/test"

VOCAB_COUNT = 200000
SAMPLE_CHUNK_SIZE = 1000

def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()

def tokenize(filename, dest_dir):
    print("Tokenizing files from {}".format(filename))
    print("Making list of files to tokenize")
    with open("mapping.txt", "w") as write_file:
        with open(filename, "r") as urls_file:
            for url in urls_file:
                story_file = hashhex(url.strip().encode("utf-8")) + ".story"
                dest_path = os.path.join(dest_dir, story_file)

                if os.path.isfile(os.path.join("data/cnn/stories/", story_file)):
                    story_file = os.path.join("data/cnn/stories/", story_file)
                elif os.path.isfile(os.path.join("data/dailymail/stories/", story_file)):
                    story_file = os.path.join("data/dailymail/stories/", story_file)
                else:
                    print("File for url {} with hash {} not found".format(url, story_file))
                    continue
                write_file.write("{} \t {}\n".format(story_file, dest_path))

    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Saving tokenozed files to {}...".format(dest_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")

def get_article_and_abstract(filename):
    article_lines = []
    abstract_lines = []

    next_line_is_abstract = False
    with open(filename, "r") ad f:
        for line in f:
            if line.strip() == "":
                continue
            elif line.startswith("@highlight"):
                next_line_is_abstract = True
            elif next_line_is_abstract == True:
                abstract_lines.append(utils.canonicalize_sentence(line))
            else:
                article_lines.append(utils.canonicalize_sentence(line))

    return " ".join(article_lines), " ".join(abstract_lines)

def write_as_tf_example_to_file(tokenized_files_dir, create_vocab=False):
    tokenized_files = os.listdir(tokenized_files_dir)

    for filename in tokenized_files:
        article, abstract = get_article_and_abstract(filename)

        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "article": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value = [article])
                    ),
                    "abstract": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value = [abstract])
                    )
                }
            )
        )

        tf_example = example.SerializeToString()

def process_cnn_dailymail_data():
    tokenize()
    write_as_tf_example_to_file()

#def process_nyt_data():
    # TODO:

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python create_datafiles.py <dataset_name(cnn-dailymail|nyt)> <article_files_dir_path>")

    articles_source = sys.argv[1]
    articles_dir = sys.argv[2]

    pathlib.Path(TOKENIZED_TRAIN_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TOKENIZED_VAL_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TOKENIZED_TEST_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TOKENIZED_SAMPLE_FILES).mkdir(parents=True, exist_ok=True)

    pathlib.Path(TF_EXAMPLE_TRAIN_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TF_EXAMPLE_VAL_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TF_EXAMPLE_TEST_FILES).mkdir(parents=True, exist_ok=True)

    if articles_source == "cnn-dailymail":
        # for files in [(TRAIN_URLS_FILES, TOKENIZED_TRAIN_FILES), (VAL_URLS_FILES, TOKENIZED_VAL_FILES), (TEST_URLS_FILES, TOKENIZED_TEST_FILES)]:
        for files in [("sample_urls.txt", TOKENIZED_SAMPLE_FILES)]:
            tokenize(files[0], files[1])
        # for files in [TOKENIZED_TRAIN_FILES, TOKENIZED_VAL_FILES, TOKENIZED_TEST_FILES]:
        for files in [TOKENIZED_SAMPLE_FILES]:
            write_as_tf_example_to_file(file)
    elif articles_source == "nyt":
        process_nyt_data()
