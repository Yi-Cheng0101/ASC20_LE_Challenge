import json, os
import torch
from modeling import AlbertForCloth
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import AlbertTokenizer

DATA_DIR = '/home/engine210/LE/dataset/ELE/test'
MODEL_DIR = '../model'
BERT_MODEL = 'albert-xxlarge-v2'
EVAL_BATCH_SIZE = 100
CACHE_SIZE = 256

def getFileList(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in sorted(file_names):
            files.append(os.path.join(root, filename))
    return files

# from data_utility
class ClothSample(object):
    def __init__(self):
        self.article = None
        self.ph = []
        self.ops = []
        self.high = 0
                    
    def convert_tokens_to_ids(self, tokenizer):
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        self.article = torch.Tensor(self.article)
        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.Tensor(self.ops[i][k])
        self.ph = torch.Tensor(self.ph)  

def tokenize_ops(ops, tokenizer):
    ret = []
    for i in range(4):
        ret.append(tokenizer.tokenize(ops[i]))
    return ret

def createSample(tokenizer, data):
    # pdb.set_trace()
    cnt = 0
    article = tokenizer.tokenize(data['article'])

    if (len(article) <= 512):
        sample = ClothSample()
        sample.article = article
        sample.high = data['high']
        for p in range(len(article)):
            if ('_' in article[p]):
                sample.article[p] = '[MASK]'
                sample.ph.append(p)
                ops = tokenize_ops(data['options'][cnt], tokenizer)
                sample.ops.append(ops)
                cnt += 1
        return [sample]
    else:
        first_sample = ClothSample()
        second_sample = ClothSample()
        first_sample.high = data['high']
        second_sample.high = data['high']
        second_s = len(article) - 512
        for p in range(len(article)):
            if ('_' in article[p]):
                article[p] = '[MASK]'
                ops = tokenize_ops(data['options'][cnt], tokenizer)
                if (p < 512):
                    first_sample.ph.append(p)
                    first_sample.ops.append(ops)
                else:
                    second_sample.ph.append(p - second_s)
                    second_sample.ops.append(ops)
                cnt += 1                    
        first_sample.article = article[:512]
        second_sample.article = article[-512:]
        if (len(second_sample.ops) == 0):
            return [first_sample]
        else:
            return [first_sample, second_sample]

def to_device(L, device):
    if (type(L) != list):
        return L.to(device)
    else:
        ret = []
        for item in L:
            ret.append(to_device(item, device))
        return ret

def preprocessor(tokenizer, file_name):
    data = json.loads(open(file_name, 'r').read())
    data['high'] = 0
    data_tensor = createSample(tokenizer, data)
    for i in range(len(data_tensor)):
        data_tensor[i].convert_tokens_to_ids(tokenizer)
    return data_tensor

class Loader(object):
    def __init__(self, data, cache_size, batch_size, device='cpu'):
        self.data = data
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.data_num = len(self.data)
        self.device = device
    
    def _batchify(self, data_set, data_batch):
        max_article_length = 0
        max_option_length = 0
        max_ops_num = 0
        bsz = len(data_batch)
        for idx in data_batch:
            data = data_set[idx]
            max_article_length = max(max_article_length, data.article.size(0))
            for ops in data.ops:
                for op in ops:
                    max_option_length = max(max_option_length, op.size(0))
            max_ops_num  = max(max_ops_num, len(data.ops))
        articles = torch.zeros(bsz, max_article_length).long()
        articles_mask = torch.ones(articles.size())
        options = torch.zeros(bsz, max_ops_num, 4, max_option_length).long()
        options_mask = torch.ones(options.size())
        answers = torch.zeros(bsz, max_ops_num).long()
        mask = torch.zeros(answers.size())
        question_pos = torch.zeros(answers.size()).long()
        for i, idx in enumerate(data_batch):
            data = data_set[idx]
            articles[i, :data.article.size(0)] = data.article
            articles_mask[i, data.article.size(0):] = 0
            for q, ops in enumerate(data.ops):
                for k, op in enumerate(ops):
                    options[i,q,k,:op.size(0)] = op
                    options_mask[i,q,k, op.size(0):] = 0
            for q, pos in enumerate(data.ph):
                question_pos[i,q] = pos
        inp = [articles, articles_mask, options, options_mask, question_pos]
        return inp
    
    def data_iter(self):
        seqlen = torch.zeros(self.data_num)
        for i in range(self.data_num):
            seqlen[i] = self.data[i].article.size(0)
        cache_start = 0
        while (cache_start < self.data_num):
            cache_end = min(cache_start + self.cache_size, self.data_num)
            cache_data = self.data[cache_start:cache_end]
            seql = seqlen[cache_start:cache_end]
            _, indices = torch.sort(seql, descending=True)
            batch_start = 0
            while (batch_start + cache_start < cache_end):
                batch_end = min(batch_start + self.batch_size, cache_end - cache_start)
                data_batch = indices[batch_start:batch_end]
                inp = self._batchify(cache_data, data_batch)
                inp = to_device(inp, self.device)
                yield inp
                batch_start += self.batch_size
            cache_start += self.cache_size

def main():
    device = torch.device("cuda:0")

    model = AlbertForCloth.from_pretrained(MODEL_DIR, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
    model.to(device)
    model.eval()

    tokenizer = AlbertTokenizer.from_pretrained(BERT_MODEL)

    file_list = getFileList(DATA_DIR)
    
    ans_dic = {0:"A", 1:"B", 2:"C", 3:"D"}
    answers = {}

    for file_path in file_list:
        file_name = file_path.split("/")[-1].replace('.json', '')
        print(file_name)
        data_tensor = preprocessor(tokenizer, file_path)
        valid_data = Loader(data_tensor, CACHE_SIZE, EVAL_BATCH_SIZE, device)

        for inp in valid_data.data_iter():
            with torch.no_grad():
                out = model(inp)
                out = out.cpu().numpy()
                answer = list(map(lambda x: ans_dic[x], out))
                print(answer)
                answers[file_name] = answer


    jsonstr = json.dumps(answers)
    jsonstr = jsonstr.replace(": ", ",").replace("], ", "],\n").replace("{", "{\n").replace("}", "\n}").replace(' ', '')

    with open('out1.json', 'w') as f:
        f.write(jsonstr)

if __name__ == "__main__":
    main()