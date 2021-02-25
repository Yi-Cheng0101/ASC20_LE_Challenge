import torch, json, re
from essential import *
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').cuda()
roberta.eval().cuda()

total_ac = 0
total_wa = 0
for t in range(1,401) :
    file_location = "/home/michael1017/NLP/ELE/dev/dev" + str(t).zfill(4) + ".json"
    with open(file_location, 'r') as r :
        data = json.load(r)
    text = split_into_sentences(data['article'])

    counter = 0
    answer = ""
    guess = True
    ac = 0
    wa = 0
    len_text = len(text)
    for i in range(len_text) :
        if "_" in text[i] :
            dash_location = [m.start() for m in re.finditer('_', text[i])]
            for j in dash_location :
                counter += 1
                cptext = text[i][:j] + text[i][j].replace('_', '<mask>') + text[i][j+1:]
                out = glue_text(len_text, cptext, roberta, text, i)
                answer = find_ans(roberta.fill_mask(out, topk=5), data['options'][counter-1], out)
                if data['answers'][counter-1] == answer:
                    ac += 1
                    total_ac += 1
                    #print(counter, "AC", answer, data['answers'][counter-1])
                    #print("===================================================================")
                else :
                    wa += 1
                    total_wa += 1
                    #print(counter, "WA", answer, data['answers'][counter-1], guess)
                    #print(roberta.fill_mask(out, topk=5))
                    #print(data['options'][counter-1][0], data['options'][counter-1][1], data['options'][counter-1][2], data['options'][counter-1][3])
                    #print("===================================================================")
    print(t, ac, "/", counter, ac/counter, total_ac /(total_ac + total_wa))
print(total_ac, "/", total_ac + total_wa, total_ac /(total_ac + total_wa))