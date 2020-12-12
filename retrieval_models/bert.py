from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


class BERTSearch(nn.Module):
    def __init__(self, cfg):
        super(BERTSearch, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.fnn = nn.Linear(768, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not cfg.get('use_cuda', True):
            self.device = 'cpu'
    def process_input(self, x):
        x = [item[0] for item in x]
        return ' '.join(x)
    def prepare_input(self, inp1, inp2):
        inp1 = self.process_input(inp1)
        inp2 = self.process_input(inp2)
        inp = f"[CLS] {inp1} [SEP] {inp2}"
        return inp
    def forward(self, sent1, sent2):
        sentence = self.prepare_input(sent1, sent2)
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        for k, v in inputs.items():
            v = torch.squeeze(v)
            w = v[:512]
            inputs[k] = torch.unsqueeze(w, 0)
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0][0]
        last_hidden_states = last_hidden_states[0]
        out = self.fnn(last_hidden_states)
        return out
