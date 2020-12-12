import pickle

import gensim.downloader as api
from gensim.models import FastText
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from models.lstm import LSTMModel
from models.transformer import TransformerModel


def elementwise_apply(fn, *args):
    return torch.nn.utils.rnn.PackedSequence(fn(*[(arg.data if type(arg)==torch.nn.utils.rnn.PackedSequence else arg) for arg in args]), args[0].batch_sizes)


def pad_input(x):
    x_lens = [len(i) for i in x]
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)

    x_pad = pack_padded_sequence(
        x_pad, x_lens, batch_first=True, enforce_sorted=False)
    return x_pad


def create_emb_layer(cfg, vocab_size, weights_matrix=None):
    input_dim = cfg.get('embedding_dim', 100)
    if weights_matrix is not None:
        emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
        non_trainable = cfg.get('freeze_embeddings', False)
        if non_trainable:
            emb_layer.weight.requires_grad = False
    else:
        emb_layer = nn.Embedding(vocab_size, input_dim)
        initrange = 0.1
        emb_layer.weight.data.uniform_(-initrange, initrange)
    return emb_layer


# model = api.load(f"glove-wiki-gigaword-100")
model = FastText.load('pretrained_embeddings/fasttext.model')
def pretrained_weights_matrix(vocab, cfg, vocab_size):
    weights_matrix = np.random.randn(vocab_size, cfg.get('embedding_dim', 100))
    for word, ix in vocab.items():
        try:
            weights_matrix[ix] = model.wv[word]
        except KeyError:
            pass
    return weights_matrix


class SentenceEncoder(nn.Module):
    def __init__(self, cfg, weights_matrix=None):
        super(SentenceEncoder, self).__init__()
        vocab_size = cfg.get('vocab_size', {})
        pretrained_embeddings = cfg.get('pretrained_embeddings', False)
        if pretrained_embeddings:
            vocab = pickle.load(open('vocab/words.pkl', 'rb'))
            weights_matrix = pretrained_weights_matrix(vocab, cfg, vocab_size.get('sentences', None))
        self.embedding = create_emb_layer(cfg, vocab_size.get('sentences', None), weights_matrix)
        self.encoder_type = cfg.get('encoder_type', 'lstm')
        if self.encoder_type not in ['lstm', 'transformer']:
            raise ValueError('Encoder needs to be valid type.')
        if self.encoder_type == 'lstm':
            self.encoder = LSTMModel(cfg)
        elif self.encoder_type == 'transformer':
            self.encoder = TransformerModel(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not cfg.get('use_cuda', True):
            self.device = torch.device('cpu')

    def forward(self, x):
        x = x.to(torch.device('cpu'))
        self.embedding.to(torch.device('cpu'))
        if self.encoder_type == 'lstm':
            # x = pad_input(x)
            output = self.embedding(x)
            # output = elementwise_apply(self.embedding, x)
            output.to(self.device)
        elif self.encoder_type == 'transformer':
            # x = pad_sequence(x, batch_first=True, padding_value=0)
            output = self.embedding(x)
        output = output.to(torch.device(self.device))
        output = self.encoder(output)
        return output


class EventEncoder(nn.Module):
    def __init__(self, cfg):
        super(EventEncoder, self).__init__()
        vocab_size = cfg.get('vocab_size', {})
        pretrained_embeddings = cfg.get('pretrained_embeddings', False)
        if pretrained_embeddings:
            trigger_vocab = pickle.load(open('vocab/triggers.pkl', 'rb'))
            argument_vocab = pickle.load(open('vocab/arguments.pkl', 'rb'))
            wm_triggers = pretrained_weights_matrix(trigger_vocab, cfg, vocab_size.get('triggers', None))
            wm_arguments = pretrained_weights_matrix(argument_vocab, cfg, vocab_size.get('arguments', None))
        else:
            wm_triggers = None
            wm_arguments = None
        self.trigger_embedding = create_emb_layer(cfg, vocab_size.get('triggers', None), wm_triggers)
        self.type_embedding = create_emb_layer(cfg, vocab_size.get('types', None), None)
        self.argument_embedding = create_emb_layer(cfg, vocab_size.get('arguments', None), wm_arguments)
        self.role_embedding = create_emb_layer(cfg, vocab_size.get('roles', None), None)
        self.encoder_type = cfg.get('encoder_type', 'lstm')
        if self.encoder_type not in ['lstm', 'transformer']:
            raise ValueError('Encoder needs to be valid type.')
        if self.encoder_type == 'lstm':
            encoder_model = LSTMModel
        elif self.encoder_type == 'transformer':
            encoder_model = TransformerModel
        self.trigger_encoder = encoder_model(cfg)
        self.type_encoder = encoder_model(cfg)
        self.argument_encoder = encoder_model(cfg)
        self.role_encoder = encoder_model(cfg)

    def forward(self, x):
        triggers, args, roles, types = x

        if self.encoder_type == 'lstm':
            # try:
            #     triggers = pad_input(triggers)
            # except:
            #     print(triggers)
            #     exit()
            # args = pad_input(args)
            # roles = pad_input(roles)
            # types = pad_input(types)

            # triggers = elementwise_apply(self.trigger_embedding, triggers)
            # args = elementwise_apply(self.argument_embedding, args)
            # roles = elementwise_apply(self.role_embedding, roles)
            # types = elementwise_apply(self.type_embedding, types)

            triggers = self.trigger_embedding(triggers)
            args = self.argument_embedding(args)
            roles = self.role_embedding(roles)
            types = self.type_embedding(types)

        elif self.encoder_type == 'transformer':
            # triggers = pad_sequence(triggers, batch_first=True, padding_value=0)
            # args = pad_sequence(args, batch_first=True, padding_value=0)
            # roles = pad_sequence(roles, batch_first=True, padding_value=0)
            # types = pad_sequence(types, batch_first=True, padding_value=0)
            triggers = self.trigger_embedding(triggers)
            args = self.argument_embedding(args)
            roles = self.role_embedding(roles)
            types = self.type_embedding(types)
        
        triggers = self.trigger_encoder(triggers)
        args = self.argument_encoder(args)
        roles = self.role_encoder(roles)
        types = self.type_encoder(types)

        output = torch.cat((triggers, args, roles, types), dim=1)
        return output