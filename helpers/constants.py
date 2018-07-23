# Constants for use by other modules.
# This module should not import any others, except for standard library.

START_TOKEN = u"<s>"
END_TOKEN   = u"</s>"
UNK_TOKEN   = u"<unk>"
UNICODE_SINGLE_CLOSE_QUOTE = u'\u2019'
UNICODE_DOUBLE_CLOSE_QUOTE = u'\u201d'

# Sentence terminators
SENTENCE_END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', UNICODE_SINGLE_CLOSE_QUOTE, UNICODE_DOUBLE_CLOSE_QUOTE, ")"]

START_DECODING = "[START]"
STOP_DECODING = "[STOP]"
PAD_TOKEN = "[PAD]"
