import os
import json
path = "/mnt/SC/ASC/task3/ELE/dev/"
files = os.listdir(path)
map_ans = {'A':0,'B':1,'C':2,'D':3}
multi_file = []

def is_multi(candidate):
    for j in candidate:
        if len(j.split()) >= 2:
            return True
    return False


# 找出 multi_file
for file in files:
    with open(path+file,"r") as f:
        data = json.load(f)
        text = data['article']
        candidates = data['options']
        answer = data['answers']
        #print(candidates)
        for candidate in candidates:
            if is_multi(candidate):
                multi_file.append(file)
                break
                
# replace single _ to correct answer
for file in multi_file:
    print(file)
    with open(path+file,"r") as f:
        data = json.load(f)
        text = data['article']
        candidates = data['options']
        answer = data['answers']
        
        ## main code
        for i in range(len(answer)):
            ##  多選先轉成 ^ ，單選直接換掉
            #print(candidates[i])
            #print(candidates[i][map_ans[answer[i]]])
            if is_multi(candidates[i]):
                data['article'] = data['article'].replace('_',"^",1)
                
            else: 
                data['article'] = data['article'].replace('_',candidates[i][map_ans[answer[i]]],1)
                data['answers'][i] = 'x'
                data['options'][i] = 'x'
                
        data['article'] = data['article'].replace('^',"_",100)
        data['answers'] = list(filter(lambda a: a != 'x', data['answers']))
        data['options'] = list(filter(lambda a: a != 'x', data['options']))
        print(text)
        
        ret = json.dumps(data)

    with open('multi'+file+'.json', 'w') as fp:
        fp.write(ret)

