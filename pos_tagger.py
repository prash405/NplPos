import os
from indicnlp.tokenize import indic_tokenize
from indicnlp import loader
import nltk
from nltk.tag import tnt
from nltk.corpus import indian


INDIC_NLP_RESOURCES = "F:/devops/nlp/pos_indian_nlp_project/indic_nlp_resources-master"
os.environ["INDIC_RESOURCES_PATH"] = INDIC_NLP_RESOURCES


loader.load()


nltk.download('indian')
nltk.download('punkt')
hindi_text = indian.tagged_sents('hindi.pos')


tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(hindi_text)


test_sentence = "राम बाजार गया।"
tokens = list(indic_tokenize.trivial_tokenize(test_sentence, lang='hi'))


tags = tnt_pos_tagger.tag(tokens)
print("POS Tags:", tags)
