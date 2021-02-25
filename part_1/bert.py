import torch
from torch import tensor
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from IPython.display import clear_output
import re
import json

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
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

PRETRAINED_MODEL_NAME = "/home/michael1017/NLP/wwm_uncased_L-24_H-1024_A-16"  # 指定繁簡中文 BERT-BASE 預訓練模型
#PRETRAINED_MODEL_NAME = "bert-large-uncased" 
# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
maskedLM_model.eval().to('cuda')
total_ac = 0
total_wa = 0
for t in range(1,400) :
    file_location = "/home/michael1017/NLP/ELE/dev/dev" + str(t).zfill(4) + ".json"
    with open(file_location, 'r') as r :
        data = json.load(r)
    text = split_into_sentences(data['article'].replace('_', '[MASK]'))

    counter = 0
    answer = ""
    guess = True
    ac = 0
    wa = 0
    len_text = len(text)
    for i in range(len_text) :
        while "[MASK]" in text[i] :
            if len_text == 1:
                target = "[CLS] " + text[i]
            elif i == 0 :
                target = "[CLS] " + text[i] + " [SEP] " +  text[i+1]
            elif i == len(text) - 1 :
                target = "[CLS] " + text[i-1] + " [SEP] " + text[i] 
            else :
                target = "[CLS] " + text[i-1] + " [SEP] " + text[i] + " [SEP] " + text[i+1]
            #print(target)
            tokens = tokenizer.tokenize(target)
            ids = tokenizer.convert_tokens_to_ids(tokens)

            tokens_tensor = torch.tensor([ids]).to('cuda') # (1, seq_len)
            segments_tensors = torch.zeros_like(tokens_tensor).to('cuda')  # (1, seq_len)

            with torch.no_grad():
                outputs = maskedLM_model(tokens_tensor, segments_tensors)
                predictions = outputs[0]

            masked_index = tokens.index("[MASK]")
            probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), 10)
            predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
            #print(predicted_tokens)
            data['options'][counter][0] = data['options'][counter][0].strip()
            data['options'][counter][1] = data['options'][counter][1].strip()
            data['options'][counter][2] = data['options'][counter][2].strip()
            data['options'][counter][3] = data['options'][counter][3].strip()

            for j in predicted_tokens:
                j = j.strip()
                if j == data['options'][counter][0]:
                    answer = "A"
                    text[i] = text[i].replace('[MASK]', data['options'][counter][0], 1)
                    break
                elif j == data['options'][counter][1]:
                    answer = "B"
                    text[i] = text[i].replace('[MASK]', data['options'][counter][1], 1)
                    break
                elif j == data['options'][counter][2]:
                    answer = "C"
                    text[i] = text[i].replace('[MASK]', data['options'][counter][2], 1)
                    break
                elif j == data['options'][counter][3]:
                    answer = "D"
                    text[i] = text[i].replace('[MASK]', data['options'][counter][3], 1)
                    break
            else :
                answer = "C"
                text[i] = text[i].replace('[MASK]', data['options'][counter][2], 1)

            if data['answers'][counter] == answer:
                ac += 1
                total_ac += 1
                #print(counter, "AC", answer, data['answers'][counter])
                #print("===================================================================")
            else :
                wa += 1
                total_wa += 1
                #print(counter, "WA", answer, data['answers'][counter], guess)
                #print(data['options'][counter][0], data['options'][counter][1], data['options'][counter][2], data['options'][counter][3])
                #print("===================================================================")

            counter += 1
    print(t, ac, "/", counter, ac/counter, total_ac /(total_ac + total_wa))
print(total_ac, "/", total_ac + total_wa, total_ac /(total_ac + total_wa))
