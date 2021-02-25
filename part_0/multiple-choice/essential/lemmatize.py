from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res

def lemmatize_text(text):
    text = text.strip()
    lemmatizer = WordNetLemmatizer()
    word_and_pos = pos_tag(word_tokenize(text))
    if len(word_and_pos) == 0:
        return text
    else:
        word, pos = word_and_pos[0]
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res = lemmatizer.lemmatize(word, pos=wordnet_pos)
        return res