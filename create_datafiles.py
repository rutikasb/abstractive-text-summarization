import os, sys
import pathlib
import subprocess
import tensorflow as tf
import hashlib
import struct
import glob
import re
import xml.etree.ElementTree

from helpers import constants, utils, vocabulary

TRAIN_URLS_FILES = "data_splits/all_train.txt"
VAL_URLS_FILES = "data_splits/all_val.txt"
TEST_URLS_FILES = "data_splits/all_test.txt"

TOKENIZED_TRAIN_FILES = "data/tokens/train"
TOKENIZED_VAL_FILES = "data/tokens/val"
TOKENIZED_TEST_FILES = "data/tokens/test"

TF_EXAMPLE_FILES_PATH = "data/bins/"
TF_EXAMPLE_TRAIN_FILE = "data/bins/train.bin"
TF_EXAMPLE_VAL_FILE = "data/bins/val.bin"
TF_EXAMPLE_TEST_FILE = "data/bins/test.bin"

NYT_ORIGINAL_FILES_PATH = "data/nyt"
NYT_STORIES_PATH = "data/nyt/stories"
NYT_TOKENS_PATH = "data/tokens/nyt"

CHUNKED_FILES_PATH = "data/chunks/"

VOCAB_COUNT = 300000
SAMPLE_CHUNK_SIZE = 1000

def hashhex(s):
  # Returns a heximal formated SHA1 hash of the input string
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()

def tokenize_cnn_dailymail(filename, dest_dir):
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
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            elif line.startswith("@highlight"):
                next_line_is_abstract = True
            elif next_line_is_abstract == True:
                abstract_lines.append(utils.canonicalize_sentence(line))
            else:
                article_lines.append(utils.canonicalize_sentence(line))

    article = " ".join(article_lines)
    abstract = " ".join(abstract_lines)
    return article, abstract

def create_cnn_like_story_files(filename, stories_dir_path):
    article_lines = []
    abstract_lines = []

    root = xml.etree.ElementTree.parse(filename).getroot()

    try:
        article_id = root.find('head').find('docdata').find('doc-id').attrib['id-string']
        if root. find('body').find('body.head').find('abstract') is not None:
            paragraphs = root. find('body').find('body.head').find('abstract').findall('p')
            for para in paragraphs:
                abstract_lines.append(para.text.replace("''", '"').replace("(M)", "").strip())

            for block in root.find('body').find('body.content').findall('block'):
                if block.attrib['class'] == 'full_text':
                    for p in block.findall('p'):
                        article_lines.append(p.text.replace("''", '"'))
            if len(article_lines) == 0 or len(abstract_lines) == 0:
                return

            with open((os.path.join(stories_dir_path, str(article_id)) +  ".xml"), "w") as f:
                for line in article_lines:
                    f.write(line + "\n\n")
                for line in abstract_lines:
                    f.write("@highlight" + "\n\n")
                    f.write(line + "\n\n")
        else:
            return
    except:
        return

def tokenize_nyt(stories_dir, tokens_dir):
    pathlib.Path(tokens_dir).mkdir(parents=True, exist_ok=True)
    print("Tokenizing files from {}".format(stories_dir))
    print("Making list of files to tokenize")
    story_files = glob.glob(os.path.join(stories_dir, "*"))
    with open("mapping.txt", "w") as write_file:
        for story_file in story_files:
            dest_file = os.path.join(tokens_dir, story_file.split("/")[-1])
            write_file.write("{} \t {}\n".format(story_file, dest_file))

    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Saving tokenozed files to {}...".format(tokens_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")

def write_as_tf_example_to_file(tokenized_files_dir, data_output_file, create_vocab=False):
    tokenized_files = os.listdir(tokenized_files_dir)

    tokens = []

    with open(data_output_file, "wb") as data_file:
        for filename in tokenized_files:
            full_file_path = os.path.join(tokenized_files_dir, filename)
            article, abstract = get_article_and_abstract(full_file_path)

            # Some data files are missing the article text
            if article == "" or abstract == "":
                print("Article or Abstract is missing for {}. Skipping".format(full_file_path))
                continue

            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        "article": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value = [bytes(article, encoding="utf-8")])
                        ),
                        "abstract": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value = [bytes(abstract, encoding="utf-8")])
                        )
                    }
                )
            )

            tf_example = example.SerializeToString()
            str_len = len(tf_example)
            data_file.write(struct.pack('q', str_len))
            data_file.write(struct.pack('%ds' % str_len, tf_example))

            if create_vocab == True:
                article_tokens =  [token for token in article.split(" ") if token not in [constants.START_TOKEN, constants.END_TOKEN]]
                abstract_tokens =  [token for token in abstract.split(" ") if token not in [constants.START_TOKEN, constants.END_TOKEN]]
                tokens += article_tokens
                tokens += abstract_tokens

    print("Finished writing examples to file {}".format(data_output_file))
    if create_vocab == True:
        print("Preparing to write vocabulary to file.")
        vocab = vocabulary.Vocabulary(tokens, size=VOCAB_COUNT)
        vocab.write_most_common_to_flat_file_with_counts("vocabulary.txt")
        print("Finished writing vocabulary to file.")


def chunk_file(filename):
    in_file = TF_EXAMPLE_FILES_PATH + filename
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(CHUNKED_FILES_PATH, '%s_%03d.bin' % (filename, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(SAMPLE_CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1
    reader.close()

def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(CHUNKED_FILES_PATH):
        os.mkdir(CHUNKED_FILES_PATH)
    # Chunk the data
    for set_name in ['train.bin', 'val.bin', 'test.bin']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % CHUNKED_FILES_PATH)

def process_cnn_dailymail_data():
    pathlib.Path(TOKENIZED_TRAIN_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TOKENIZED_VAL_FILES).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TOKENIZED_TEST_FILES).mkdir(parents=True, exist_ok=True)

    pathlib.Path("data/bins").mkdir(parents=True, exist_ok=True)

    for files in [(TRAIN_URLS_FILES, TOKENIZED_TRAIN_FILES), (VAL_URLS_FILES, TOKENIZED_VAL_FILES), (TEST_URLS_FILES, TOKENIZED_TEST_FILES)]:
        tokenize_cnn_dailymail(files[0], files[1])
    for files in [(TOKENIZED_TRAIN_FILES, TF_EXAMPLE_TRAIN_FILE),
            (TOKENIZED_VAL_FILES, TF_EXAMPLE_VAL_FILE),
            (TOKENIZED_TEST_FILES, TF_EXAMPLE_TEST_FILE)]:
        matcher = re.compile(r".*train.*", re.IGNORECASE)
        create_vocab = True if matcher.match(files[0]) else False
        write_as_tf_example_to_file(files[0], files[1], create_vocab)
    chunk_all()

def process_nyt_data():
    files = glob.glob(os.path.join(NYT_ORIGINAL_FILES_PATH, "**/**/**/*.xml"))
    # files = glob.glob(os.path.join(NYT_ORIGINAL_FILES_PATH, "2001/01/**/*.xml"))

    pathlib.Path(NYT_STORIES_PATH).mkdir(parents=True, exist_ok=True)

    for filename in files:
        create_cnn_like_story_files(filename, NYT_STORIES_PATH)

    tokenize_nyt(NYT_STORIES_PATH, NYT_TOKENS_PATH)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python create_datafiles.py <dataset_name(cnn-dailymail|nyt)>")

    articles_source = sys.argv[1]

    if articles_source == "cnn-dailymail":
        process_cnn_dailymail_data()
    elif articles_source == "nyt":
        process_nyt_data()
