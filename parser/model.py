# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

from parser.modules import crfae

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        # the embedding layer
        if args.bert is False:
            self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                        embedding_dim=args.word_embed)
            if args.freeze_word_emb:
                self.word_embed.weight.requires_grad = False
        else:
            self.word_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.word_embed)

        self.feat_embed = nn.Embedding(num_embeddings=args.n_feats,
                                        embedding_dim=args.n_embed)

        if args.freeze_feat_emb:
            self.feat_embed.weight.requires_grad = False
        
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=args.word_embed+args.n_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc,
                             dropout=args.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

        self.multinomial = nn.Parameter(torch.ones(args.n_feats, args.n_feats))


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def W_Reg(self):
        diff = 0.
        for name, param in self.named_parameters():
            if name == 'pretrained.weight':
                continue
            diff += ((self.init_params[name] - param) ** 2).sum()

        return 0.5 * diff * self.args.W_beta

    def E_Reg(self, words, bert, feats, source_model, tar_score):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        diff = ((source_score - tar_score) ** 2).sum()
        return 0.5 * diff * self.args.E_beta

    def T_Reg(self, words, bert, feats, source_model):
        source_model.eval()
        with torch.no_grad():
            source_score = source_model(words, bert, feats)
            source_score = source_model.decoder(source_score, feats)
        return source_score

    def forward(self, words, bert, feats):
        self.batch_size, self.seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)

        if self.args.bert is False:
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)
            word_embed = self.word_embed(ext_words)
        else:
            word_embed = self.word_embed(*bert)


        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[mask])
            feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
        else:
            feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        embed = torch.cat((word_embed, feat_embed), dim=-1)

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=self.seq_len)
        x = self.lstm_dropout(x)

        arc_h = self.mlp_arc_h(x)  # [batch_size, seq_len, d]
        arc_d = self.mlp_arc_d(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        return s_arc

    def decoder(self, crf_weight, feats):
        m = nn.Softmax(dim=1)(self.multinomial)
        recons_weight = torch.log(m[feats[:, :, None], feats[:, None, :]])  # B * N * N
        joint_weights = crf_weight + recons_weight
        return joint_weights

    def crf(self, crf_weights, joint_weights, heads, words, pos):
        if self.args.unsupervised:
            if self.args.paskin:
                f = crfae.CRFAE(self.seq_len, self.batch_size, self.args)
                log_partition, best_score = f(crf_weights, joint_weights)
                self.best_tree = f.best_tree
                return -(best_score - log_partition).mean() 
            else:
                heads = joint_weights.argmax(1)

        crf_weights = crf_weights[:, :, 1:]
        crf_weights = torch.logsumexp(crf_weights, dim=1)
        log_partition = crf_weights.sum(dim=1)

        sent_idx = torch.arange(self.batch_size, device=self.args.device).contiguous().view(-1, 1).long()
        children_token_id = torch.arange(self.seq_len, device=self.args.device).contiguous().view(1, -1).long()
        best_score = joint_weights[sent_idx, heads[:, 1:], children_token_id[:, 1:]]
        best_score = best_score.sum(dim=1)

        loss = -(best_score - log_partition)
        loss = loss.mean()

        return loss

    def decode(self, arc_scores, mask):
        if self.args.tree:
            arc_preds = eisner(arc_scores, mask)
        else:
            arc_preds = arc_scores.argmax(-1)

        return arc_preds

    def decode_crf(self, arc_scores, mask):
        arc_preds = arc_scores.argmax(1)
        return arc_preds

    def decode_paskin(self, arc_scores):
        f = crfae.CRFAE(self.seq_len, self.batch_size, self.args)
        best_score, best_tree = f.decoding(arc_scores)
        return best_tree


    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
