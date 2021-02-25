import torch
from transformers import RobertaTokenizer, RobertaForMultipleChoice

# configuration
PRETRAINED_MODEL_NAME = "/mnt/shared/engine210/LE/model/swag/roberta-large-e10"
roberta_tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
roberta_model = RobertaForMultipleChoice.from_pretrained(PRETRAINED_MODEL_NAME)
roberta_model.eval().to('cuda:1')

def find_mulit_choice(long_text, options) :
    choices = []
    temp_len = []
    for i in options:
        choices.append(long_text.replace('<mask>', i))

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
    
    return max_num_index

def find_ans(long_text, options):
    # long_text is a string
    # predictions contains 5 ans
    options = list(map(str.strip, options)) # remove the space in front and behind the options

    return find_mulit_choice(long_text, options)