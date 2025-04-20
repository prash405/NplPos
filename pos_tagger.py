# Install required libraries first
# pip install indic-nlp-library
# pip install nltk

from indicnlp.tokenize import indic_tokenize
from indicnlp import loader
import nltk
from nltk.tag import tnt
from nltk.corpus import indian

# Load Indic NLP resources
INDIC_NLP_RESOURCES = './indic_nlp_resources'  # adjust path if needed
loader.load()

# Load Hindi corpus from NLTK (or use your own tagged corpus)
nltk.download('indian')
nltk.download('punkt')
hindi_text = indian.tagged_sents('hindi.pos')

# Train a POS tagger
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(hindi_text)

# Sample Hindi sentence
test_sentence = "राम बाजार गया।"
tokens = list(indic_tokenize.trivial_tokenize(test_sentence, lang='hi'))

# Predict POS tags
tags = tnt_pos_tagger.tag(tokens)
print("POS Tags:", tags)
