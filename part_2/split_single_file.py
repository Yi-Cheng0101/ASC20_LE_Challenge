import os
import json
path = "/mnt/SC/ASC/task3/ELE/dev/"
files = os.listdir(path)
map_ans = {'A':0,'B':1,'C':2,'D':3}
single_file = []

def is_multi(candidate):
    for j in candidate:
        if len(j.split()) >= 2:
            return True
    return False


# 找出 single_file
for file in files:
    with open(path+file,"r") as f:
        data = json.load(f)
        text = data['article']
        candidates = data['options']
        answer = data['answers']
        #print(candidates)
        for candidate in candidates:
            if is_multi(candidate):
                pass
            else:
                single_file.append(file)
                break
                
# replace single _ to correct answer
for file in single_file:
    print(file)
    with open(path+file,"r") as f:
        data = json.load(f)
        text = data['article']
        candidates = data['options']
        answer = data['answers']
        
        ## main code
        for i in range(len(answer)):
            if is_multi(candidates[i]):
                print(data['options'][i] )
                print( data['answers'][i])
                print(candidates[i][map_ans[answer[i]]])
                data['article'] = data['article'].replace('_',candidates[i][map_ans[answer[i]]],1)
                data['answers'][i] = 'x'
                data['options'][i] = 'x'
            else: 
                data['article'] = data['article'].replace('_',"^",1)   
                
        data['article'] = data['article'].replace('^',"_",100)
        data['answers'] = list(filter(lambda a: a != 'x', data['answers']))
        data['options'] = list(filter(lambda a: a != 'x', data['options']))
        print(data)
        
        ret = json.dumps(data)

    with open('single'+file+'.json', 'w') as fp:
        fp.write(ret)

