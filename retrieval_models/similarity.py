import torch
import torch.nn as nn

from models.encoder import SentenceEncoder, EventEncoder


class Similarity(nn.Module):
    def __init__(self, cfg, weights_matrix=None):
        super(Similarity, self).__init__()
        if cfg['use_events']:
            self.use_events = True
            self.event_encoder = EventEncoder(cfg)
        else:
            self.use_events = False
        self.sentence_encoder = SentenceEncoder(cfg, weights_matrix)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, question, technote):
        question_sentence, question_events = question
        technote_sentence, technote_events = technote
        question_sentence = self.sentence_encoder(question_sentence)
        technote_sentence = self.sentence_encoder(technote_sentence)
        
        if self.use_events:
            question_events = self.event_encoder(question_events)
            technote_events = self.event_encoder(technote_events)

            question = torch.cat((question_sentence, question_events), dim=1)
            technote = torch.cat((technote_sentence, technote_events), dim=1)

        else:
            question = question_sentence
            technote = technote_sentence
        
        out = self.cos(question, technote)
        return out
