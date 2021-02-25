import json, re, os
from essential.sentence import SplitIntoSentences, ExpendText
from findans.xlnet import find_ans

DATA_DIR = '/home/engine210/LE/dataset/split_dev/multi_dev/'
map_num_to_ans = {0:"A", 1:"B", 2:"C", 3:"D"}

total_ac = 0
total_wa = 0

entries = os.listdir(DATA_DIR)
for idx, entry in enumerate(entries):
    with open(DATA_DIR + entry, 'r') as f:
        data = json.loads(f.read())
    
    sentences = SplitIntoSentences(data['article'])
    sentences_num = len(sentences)
    question_number = 0
    ac = 0
    wa = 0
    for i in range(sentences_num):
        while sentences[i].find('_') != -1:
            mask_sentence = sentences[i].replace('_', '<mask>', 1)
            out = ExpendText(sentences_num, mask_sentence, sentences, i)
            answer = find_ans(out, data['options'][question_number])
            sentences[i] = sentences[i].replace('_', data['options'][question_number][answer], 1)
            if data['answers'][question_number] == map_num_to_ans[answer]:
                ac += 1
                total_ac += 1
            else:
                wa += 1
                total_wa += 1
            question_number += 1

    print(idx, ac, "/", question_number, ac/question_number, total_ac /(total_ac + total_wa))
print(total_ac, "/", total_ac + total_wa, total_ac /(total_ac + total_wa))