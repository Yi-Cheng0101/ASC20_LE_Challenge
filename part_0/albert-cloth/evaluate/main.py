import json, os
import torch
from modeling import AlbertForCloth
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import AlbertTokenizer

DATA_DIR = '/home/engine210/LE/dataset/ELE/dev'
MODEL_DIR = '/mnt/shared/engine210/LE/model/albert-cloth'
BERT_MODEL = 'albert-xxlarge-v2'
EVAL_BATCH_SIZE = 100
CACHE_SIZE = 256

def getFileList(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in sorted(file_names):
            files.append(os.path.join(root, filename))
    return files

def solve(fileName):
    pass

# from data_utility
class ClothSample(object):
    def __init__(self):
        self.article = None
        self.ph = []
        self.ops = []
        self.ans = []
        self.high = 0
                    
    def convert_tokens_to_ids(self, tokenizer):
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        self.article = torch.Tensor(self.article)
        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.Tensor(self.ops[i][k])
        self.ph = torch.Tensor(self.ph)
        self.ans = torch.Tensor(self.ans)    

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
                sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
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
                    first_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                else:
                    second_sample.ph.append(p - second_s)
                    second_sample.ops.append(ops)
                    second_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                cnt += 1                    
        first_sample.article = article[:512]
        second_sample.article = article[-512:]
        if (len(second_sample.ans) == 0):
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
        high_mask = torch.zeros(bsz) #indicate the sample belong to high school set
        for i, idx in enumerate(data_batch):
            data = data_set[idx]
            articles[i, :data.article.size(0)] = data.article
            articles_mask[i, data.article.size(0):] = 0
            for q, ops in enumerate(data.ops):
                for k, op in enumerate(ops):
                    options[i,q,k,:op.size(0)] = op
                    options_mask[i,q,k, op.size(0):] = 0
            for q, ans in enumerate(data.ans):
                answers[i,q] = ans
                mask[i,q] = 1
            for q, pos in enumerate(data.ph):
                question_pos[i,q] = pos
            high_mask[i] = data.high
        inp = [articles, articles_mask, options, options_mask, question_pos, mask, high_mask]
        tgt = answers
        return inp, tgt
    
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
                inp, tgt = self._batchify(cache_data, data_batch)
                inp = to_device(inp, self.device)
                tgt = to_device(tgt, self.device)
                yield inp, tgt
                batch_start += self.batch_size
            cache_start += self.cache_size

def main():
    device = torch.device("cuda:0")
    n_gpu = 1

    model = AlbertForCloth.from_pretrained(MODEL_DIR, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
    model.to(device)
    model.eval()

    tokenizer = AlbertTokenizer.from_pretrained(BERT_MODEL)

    file_list = getFileList(DATA_DIR)

    eval_loss, eval_accuracy, eval_h_acc, eval_m_acc = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples, nb_eval_h_examples = 0, 0, 0
    for file_name in file_list:
        print(file_name)
        data_tensor = preprocessor(tokenizer, file_name)
        valid_data = Loader(data_tensor, CACHE_SIZE, EVAL_BATCH_SIZE, device)

        for inp, tgt in valid_data.data_iter():
            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy, tmp_h_acc, tmp_m_acc = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean() # mean() to average on multi-gpu.
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
                tmp_h_acc = tmp_h_acc.sum()
                tmp_m_acc = tmp_m_acc.sum()

            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            eval_h_acc += tmp_h_acc.item()
            eval_m_acc += tmp_m_acc.item()
            nb_eval_examples += inp[-2].sum().item()
            nb_eval_h_examples += (inp[-2].sum(-1) * inp[-1]).sum().item()
            nb_eval_steps += 1
        
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    if nb_eval_h_examples != 0:
        eval_h_acc = eval_h_acc / nb_eval_h_examples
    else:
        eval_h_acc = 0
    eval_m_acc = eval_m_acc / (nb_eval_examples - nb_eval_h_examples)
    result = {
        'dev_eval_loss': eval_loss,
        'dev_eval_accuracy': eval_accuracy,
        'dev_h_acc':eval_h_acc,
        'dev_m_acc':eval_m_acc}

    print(result)


if __name__ == "__main__":
    main()