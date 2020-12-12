import torch
import torch.nn as nn

from models.similarity import Similarity
from models.bert import BERTSearch


class EventBert(nn.Module):
    def __init__(self, cfg):
        super(EventBert, self).__init__()
        self.similarity = Similarity(cfg)
        self.bert = BERTSearch(cfg)
        self.fnn = nn.Linear(2, 1)
    def forward(self, question, technote, question_bert, technote_bert):
        score1 = self.similarity(question, technote)
        score2 = self.bert(question_bert, technote_bert)
        score = self.fnn(torch.cat((score1, score2)))
        return score
