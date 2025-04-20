import os
from indicnlp.tokenize import indic_tokenize
from indicnlp import loader
import nltk
from nltk.tag import tnt
from nltk.corpus import indian

# ✅ Set correct path to the resource folder
INDIC_NLP_RESOURCES = "F:/devops/nlp/pos_indian_nlp_project/indic_nlp_resources-master"
os.environ["INDIC_RESOURCES_PATH"] = INDIC_NLP_RESOURCES

# Load Indic NLP resources
loader.load()

# Load Hindi POS-tagged corpus
nltk.download('indian')
nltk.download('punkt')
hindi_text = indian.tagged_sents('hindi.pos')

# Train the tagger
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(hindi_text)

# Sample test
test_sentence = "राम बाजार गया।"
tokens = list(indic_tokenize.trivial_tokenize(test_sentence, lang='hi'))

# Get POS tags
tags = tnt_pos_tagger.tag(tokens)
print("POS Tags:", tags)
