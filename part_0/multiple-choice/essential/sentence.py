import re

# parameters for SplitIntoSentences
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

# parameters for ExpendText
MAX_LEN_FOR_SENTENCE = 128

def SplitIntoSentences(text):
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

def ExpendText(sentences_num, mask_sentence, sentences, i):
    len_words = len(mask_sentence.split()) # how many words in the sentence
    append_back = True
    append_front = True
    for j in range(1, 128):
        append_success = False
        if append_front == True and i - j >= 0:
            add_len_words = len(sentences[i-j].split())
            if len_words + add_len_words < MAX_LEN_FOR_SENTENCE:
                mask_sentence = sentences[i-j] + mask_sentence
                len_words += add_len_words
                append_success = True
            else :
                append_front = False
        if append_back == True and i + j < sentences_num:
            add_len_words = len(sentences[i+j].split())
            if len_words + add_len_words < MAX_LEN_FOR_SENTENCE:
                mask_sentence = mask_sentence + sentences[i+j]
                len_words += add_len_words
                append_success = True
            else :
                append_back = False
        if append_success == False:
            break
    return mask_sentence