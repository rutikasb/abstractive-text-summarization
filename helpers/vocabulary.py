from __future__ import print_function
from __future__ import division

from collections import defaultdict, Counter

from . import constants

class Vocabulary(object):

    START_TOKEN = constants.START_TOKEN
    END_TOKEN   = constants.END_TOKEN
    UNK_TOKEN   = constants.UNK_TOKEN

    def __init__(self, tokens, size=None,
                 progressbar=lambda l:l):
        """Create a Vocabulary object.

        Args:
            tokens: iterator( string )
            size: None for unlimited, or int > 0 for a fixed-size vocab.
                  Vocabulary size includes special tokens <s>, </s>, and <unk>
            progressbar: (optional) progress bar to wrap iterator.
        """
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(lambda: Counter())
        prev_word = None
        for word in progressbar(tokens):  # Make a single pass through tokens
            self.unigram_counts[word] += 1
            self.bigram_counts[prev_word][word] += 1
            prev_word = word
        self.bigram_counts.default_factory = None  # make into a normal dict

        # Leave space for "<s>", "</s>", and "<unk>"
        top_counts = self.unigram_counts.most_common(None if size is None else (size - 3))
        vocab = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN] +
                 [w for w,c in top_counts])

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}
        self.size = len(self.id_to_word)
        if size is not None:
            assert(self.size <= size)

        # For convenience
        self.wordset = set(self.word_to_id.keys())

        # Store special IDs
        self.START_ID = self.word_to_id[self.START_TOKEN]
        self.END_ID = self.word_to_id[self.END_TOKEN]
        self.UNK_ID = self.word_to_id[self.UNK_TOKEN]

    @property
    def num_unigrams(self):
        return len(self.unigram_counts)

    @property
    def num_bigrams(self):
        return len(self.bigram_counts)

    def __contains__(self, key):
        if isinstance(key, int):
            return (key > 0 and key < self.size)
        else:
            return key in self.word_to_id

    def words_to_ids(self, words):
        return [self.word_to_id.get(w, self.UNK_ID) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word[i] for i in ids]

    def pad_sentence(self, words, use_eos=True):
        ret = [self.START_TOKEN] + words
        if use_eos:
          ret.append(self.END_TOKEN)
        return ret

    def sentence_to_ids(self, words, use_eos=True):
        return self.words_to_ids(self.pad_sentence(words, use_eos))

    def ordered_words(self):
        """Return a list of words, ordered by id."""
        return self.ids_to_words(range(self.size))

    def write_flat_file(self, filename):
        """Write the vocabulary list to a flat file."""
        ordered_words = self.ids_to_words(range(self.size))
        with open(filename, 'w') as fd:
            for word in ordered_words:
                fd.write(word + "\n")
        print("Vocabulary ({:,} words) written to '{:s}'".format(len(ordered_words),
                                                               filename))

    def write_most_common_to_flat_file_with_counts(self, filename, count=None):
        """Write the vocabulary list to a flat file."""
        with open(filename, 'w') as fd:
            for word in self.unigram_counts.most_common(count):
                fd.write(word[0] + " " + str(word[1]) + "\n")
        print("Vocabulary ({:,} words) written to '{:s}'".format(len(self.unigram_counts.most_common(count)),
                                                               filename))

    def write_projector_config(self, checkpoint_dir, tensor_name):
        """Write metadata for TensorBoard Embeddings Projector."""
        import os
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        metadata_file = os.path.join(checkpoint_dir, "metadata.tsv")
        self.write_flat_file(metadata_file)
        # Write projector config pb
        projector_config_file = os.path.join(checkpoint_dir,
                                             "projector_config.pbtxt")
        with open(projector_config_file, 'w') as fd:
            contents = """embeddings {
              tensor_name: "%s"
              metadata_path: "metadata.tsv"
            }""" % tensor_name
            fd.write(contents)
        print("Projector config written to {:s}".format(projector_config_file))


class Vocabulary2(object):

    START_TOKEN = constants.START_TOKEN
    END_TOKEN   = constants.END_TOKEN
    UNK_TOKEN   = constants.UNK_TOKEN

    def __init__(self, vocab_filename, size=None):
        self.unigram_counts = Counter()

        if size is None:
             with open(vocab_filename, "r") as f:
                for line in f:
                    word, word_size = line.split()
                    self.unigram_counts[word] += int(word_size)
        else:
            f = open(vocab_filename, "r")
            for i in range(size):
                word, word_size = next(f).split()
                self.unigram_counts[word] += int(word_size)
            f.close()

        # Leave space for "<s>", "</s>", "<unk>", "[START]", "[STOP]", and "[PAD]"
        top_counts = self.unigram_counts.most_common(None if size is None else (size - 6))
        vocab = ([self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
            constants.START_DECODING, constants.STOP_DECODING,
            constants.PAD_TOKEN] +
                 [w for w,c in top_counts])

        # Assign an id to each word, by frequency
        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {v:k for k,v in self.id_to_word.items()}
        self.size = len(self.id_to_word)
        if size is not None:
            assert(self.size <= size)

        # For convenience
        self.wordset = set(self.word_to_id.keys())

        # Store special IDs
        self.START_ID = self.word_to_id[self.START_TOKEN]
        self.END_ID = self.word_to_id[self.END_TOKEN]
        self.UNK_ID = self.word_to_id[self.UNK_TOKEN]

    # @property
    def num_unigrams(self):
        return len(self.unigram_counts)

    def words_to_ids(self, words):
        return [self.word_to_id.get(w, self.UNK_ID) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word[i] for i in ids]

    def get_word_to_id(self, word):
        return self.word_to_id.get(word, self.UNK_ID)

    def get_id_to_word(self, i):
        return self.id_to_word[i]

    def article_to_ids(self, article_words):
        ids = []
        oov_words = []
        for word in article_words:
            i = self.word_to_id.get(word, self.UNK_ID)
            if i == self.UNK_ID:
                oov_words.append(word) if word
                oov_id = oov_words.index(word)
                ids.append(self.size + oov_id)
            else:
                ids.append(i)
        return ids, oov_words

    def abstract_to_ids(self, abstract_words, article_oov_words):
        ids = []
        for word in abstract_words:
            i = self.word_to_id.get(word, self.UNK_ID)
            if i == self.UNK_ID:
                if word in article_oov_words:
                    vocab_index = self.size + article_oov_words.index(word)
                    ids.append(vocab_index)
                else:
                    ids.append(i)
            else:
                ids.append(i)
        return ids

    def output_ids_to_words(ids, article_oov_words):
        words = []
        for i in ids:
            try:
                word = self.id_to_word[i]
            except KeyError as e:
                assert article_oov_words is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
                article_oov_index = i - self.size
                try:
                    word = article_oov_words[article_oov_index]
                except KeyError as e:
                    raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_index, len(article_oov_words)))
            words.append(word)
        return words
