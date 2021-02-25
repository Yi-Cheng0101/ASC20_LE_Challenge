from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re
from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
MAX_LEN_FOR_SENTENCE = 256
PRETRAINED_MODEL_NAME = "/mnt/shared/michael1017/NLP/roberta_model/swag_base"
roberta_tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
roberta_model = RobertaForMultipleChoice.from_pretrained(PRETRAINED_MODEL_NAME)
roberta_model.eval().to('cuda:1')
map_num_to_ans = {0:"A", 1:"B", 2:"C", 3:"D"}

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


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

def glue_text(len_text, cptext, roberta, text, i):
    len_words = len(cptext.split()) # how many words in the sentence
    append_back = True
    append_front = True
    for j in range(1, 128):
        append_success = False
        if append_front == True and i - j >= 0:
            add_len_words = len(text[i-j].split())
            if len_words + add_len_words < MAX_LEN_FOR_SENTENCE:
                cptext = text[i-j] + cptext
                len_words += add_len_words
                append_success = True
            else :
                append_front = False
        if append_back == True and i + j < len_text:
            add_len_words = len(text[i+j].split())
            if len_words + add_len_words < MAX_LEN_FOR_SENTENCE:
                cptext = cptext + text[i+j]
                len_words += add_len_words
                append_success = True
            else :
                append_back = False
        if append_success == False:
            break
    return cptext
    

def find_mask_match(text, options):
    options = list(map(str.strip, options))
    find = True
    for myans in [x[2] for x in text]:
        myans = myans.strip()
        if myans == options[0]:
            answer = "A"
            break
        elif myans == options[1] :
            answer = "B"
            break
        elif myans == options[2] :
            answer = "C"
            break
        elif myans == options[3] :
            answer = "D"
            break
    else :
        answer = "C"
        find = False
    return [answer, find]
    
def find_mask_lemmatize_match(text, options):
    options = list(map(lemmatize_text, options))
    find = True
    for myans in [x[2] for x in text]:
        myans = lemmatize_text(myans)
        if myans == options[0]:
            answer = "A"
            break
        elif myans == options[1] :
            answer = "B"
            break
        elif myans == options[2] :
            answer = "C"
            break
        elif myans == options[3] :
            answer = "D"
            break
    else :
        answer = "C"
        find = False
    return [answer, find]

def find_mulit_choice(long_text, options) :
    choices = []
    temp_len = []
    for i in range(len(options)):
        choices.append(long_text.replace('<mask>', options[i]))

    temp = [roberta_tokenizer.encode(s) for s in choices]

    for i in temp:
        temp_len.append(len(i))
    max_temp_len = max(temp_len)

    for i in range(len(temp)):
        for k in range(max_temp_len - temp_len[i]):
            temp[i].insert(1, 1)
    #print(temp)
    input_ids = torch.tensor(temp).unsqueeze(0).to('cuda:1')  # Batch size 1, 2 choices
    labels = torch.tensor(1).unsqueeze(0).to('cuda:1')  # Batch size 1

    outputs = roberta_model(input_ids, labels=labels)
    loss, classification_scores = outputs[:2]

    sfm_classification_scores = torch.softmax(classification_scores, 1)
    max_num_index = 0
    max_num = 0
    for i in range(4):
        if sfm_classification_scores[0][i] > max_num:
            max_num = sfm_classification_scores[0][i]
            max_num_index = i

    #print("!!!!!!!!!!!!!!!!!!", map_num_to_ans[max_num_index])
    return map_num_to_ans[max_num_index]


def find_ans(text, options, long_text):
    # long_text is a string
    # text contains 5 ans
    '''
    options = list(map(str.strip, options))
    r_ans, is_find = find_mask_match(text, options)
    if is_find == True:
        return r_ans

    r_ans, is_find = find_mask_lemmatize_match(text, options)
    if is_find == True:
        return r_ans
    '''
    return find_mulit_choice(long_text, options)
    