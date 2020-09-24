# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import eisner
from parser.utils.common import bos, pad, unk
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, CharField, Field
from parser.utils.fn import ispunct
from parser.utils.metric import Metric

import torch
from torch import autograd
import torch.nn as nn
from transformers import BertTokenizer

import logging


class CMD(object):

    def __call__(self, args):
        self.args = args
        logging.basicConfig(filename=args.output, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        
        args.ud_dataset = {
                'en': (
                    'data/ud/UD_English-EWT/en_ewt-ud-train.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-dev.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-test.conllx',
                    "data/fastText_data/wiki.en.ewt.vec.new",
                ),
                'en20': (
                    'data/ud/UD_English-EWT/en_ewt-ud-train20.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-dev.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-test.conllx',
                    "data/fastText_data/wiki.en.ewt.vec.new",
                ),
                'en40': (
                    'data/ud/UD_English-EWT/en_ewt-ud-train40.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-dev.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-test.conllx',
                    "data/fastText_data/wiki.en.ewt.vec.new",
                ),
                'en60': (
                    'data/ud/UD_English-EWT/en_ewt-ud-train60.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-dev.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-test.conllx',
                    "data/fastText_data/wiki.en.ewt.vec.new",
                ),
                'en80': (
                    'data/ud/UD_English-EWT/en_ewt-ud-train80.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-dev.conllx',
                    'data/ud/UD_English-EWT/en_ewt-ud-test.conllx',
                    "data/fastText_data/wiki.en.ewt.vec.new",
                ),
                'ar': (
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-train.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-dev.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-test.conllx",
                    "data/fastText_data/wiki.ar.padt.vec.new",
                ),
                'ar20': (
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-train20.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-dev.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-test.conllx",
                    "data/fastText_data/wiki.ar.padt.vec.new",
                ),
                'ar40': (
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-train40.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-dev.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-test.conllx",
                    "data/fastText_data/wiki.ar.padt.vec.new",
                ),
                'ar60': (
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-train60.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-dev.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-test.conllx",
                    "data/fastText_data/wiki.ar.padt.vec.new",
                ),
                'ar80': (
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-train80.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-dev.conllx",
                    "data/ud/UD_Arabic-PADT/ar_padt-ud-test.conllx",
                    "data/fastText_data/wiki.ar.padt.vec.new",
                ),
                'bg': (
                    "data/ud/UD_Bulgarian-BTB/bg_btb-ud-train.conllx",
                    "data/ud/UD_Bulgarian-BTB/bg_btb-ud-dev.conllx",
                    "data/ud/UD_Bulgarian-BTB/bg_btb-ud-test.conllx",
                    "data/fastText_data/wiki.bg.btb.vec.new",
                ),
                'da': (
                    "data/ud/UD_Danish-DDT/da_ddt-ud-train.conllx",
                    "data/ud/UD_Danish-DDT/da_ddt-ud-dev.conllx",
                    "data/ud/UD_Danish-DDT/da_ddt-ud-test.conllx",
                    "data/fastText_data/wiki.da.ddt.vec.new",
                ),
                'de': (
                    "data/ud/UD_German-GSD/de_gsd-ud-train.conllx",
                    "data/ud/UD_German-GSD/de_gsd-ud-dev.conllx",
                    "data/ud/UD_German-GSD/de_gsd-ud-test.conllx",
                    "data/fastText_data/wiki.de.gsd.vec.new",
                ),
                'es': (
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-train.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-dev.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-test.conllx",
                    "data/fastText_data/wiki.es.gsdancora.vec.new",
                ),
                'es20': (
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-train20.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-dev.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-test.conllx",
                    "data/fastText_data/wiki.es.gsdancora.vec.new",
                ),
                'es40': (
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-train40.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-dev.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-test.conllx",
                    "data/fastText_data/wiki.es.gsdancora.vec.new",
                ),
                'es60': (
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-train60.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-dev.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-test.conllx",
                    "data/fastText_data/wiki.es.gsdancora.vec.new",
                ),
                'es80': (
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-train80.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-dev.conllx",
                    "data/ud/UD_Spanish-GSDAnCora/es_gsdancora-ud-test.conllx",
                    "data/fastText_data/wiki.es.gsdancora.vec.new",
                ),
                'fa': (
                    "data/ud/UD_Persian-Seraji/fa_seraji-ud-train.conllx",
                    "data/ud/UD_Persian-Seraji/fa_seraji-ud-dev.conllx",
                    "data/ud/UD_Persian-Seraji/fa_seraji-ud-test.conllx",
                    "data/fastText_data/wiki.fa.seraji.vec.new",
                ),
                'fr': (
                    "data/ud/UD_French-GSD/fr_gsd-ud-train.conllx",
                    "data/ud/UD_French-GSD/fr_gsd-ud-dev.conllx",
                    "data/ud/UD_French-GSD/fr_gsd-ud-test.conllx",
                    "data/fastText_data/wiki.fr.gsd.vec.new",
                ),
                'he': (
                    "data/ud/UD_Hebrew-HTB/he_htb-ud-train.conllx",
                    "data/ud/UD_Hebrew-HTB/he_htb-ud-dev.conllx",
                    "data/ud/UD_Hebrew-HTB/he_htb-ud-test.conllx",
                    "data/fastText_data/wiki.he.htb.vec.new",
                ),
                'hi': (
                    "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllx",
                    "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllx",
                    "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllx",
                    "data/fastText_data/wiki.hi.hdtb.vec.new",
                ),
                'hr': (
                    "data/ud/UD_Croatian-SET/hr_set-ud-train.conllx",
                    "data/ud/UD_Croatian-SET/hr_set-ud-dev.conllx",
                    "data/ud/UD_Croatian-SET/hr_set-ud-test.conllx",
                    "data/fastText_data/wiki.hr.set.vec.new",
                ),
                'id': (
                    "data/ud/UD_Indonesian-GSD/id_gsd-ud-train.conllx",
                    "data/ud/UD_Indonesian-GSD/id_gsd-ud-dev.conllx",
                    "data/ud/UD_Indonesian-GSD/id_gsd-ud-test.conllx",
                    "data/fastText_data/wiki.id.gsd.vec.new",
                ),
                'it': (
                    "data/ud/UD_Italian-ISDT/it_isdt-ud-train.conllx",
                    "data/ud/UD_Italian-ISDT/it_isdt-ud-dev.conllx",
                    "data/ud/UD_Italian-ISDT/it_isdt-ud-test.conllx",
                    "data/fastText_data/wiki.it.isdt.vec.new",
                ),
                'ja': (
                    "data/ud/UD_Japanese-GSD/ja_gsd-ud-train.conllx",
                    "data/ud/UD_Japanese-GSD/ja_gsd-ud-dev.conllx",
                    "data/ud/UD_Japanese-GSD/ja_gsd-ud-test.conllx",
                    "data/fastText_data/wiki.ja.gsd.vec.new",
                ),
                'ko': (
                    "data/ud/UD_Korean-GSDKaist/ko_gsdkaist-ud-train.conllx",
                    "data/ud/UD_Korean-GSDKaist/ko_gsdkaist-ud-dev.conllx",
                    "data/ud/UD_Korean-GSDKaist/ko_gsdkaist-ud-test.conllx",
                    "data/fastText_data/wiki.ko.gsdkaist.vec.new",
                ),
                'nl': (
                    "data/ud/UD_Dutch-AlpinoLassySmall/nl_alpinolassysmall-ud-train.conllx",
                    "data/ud/UD_Dutch-AlpinoLassySmall/nl_alpinolassysmall-ud-dev.conllx",
                    "data/ud/UD_Dutch-AlpinoLassySmall/nl_alpinolassysmall-ud-test.conllx",
                    "data/fastText_data/wiki.nl.alpinolassysmall.vec.new",
                ),
                'no': (
                    "data/ud/UD_Norwegian-BokmaalNynorsk/no_bokmaalnynorsk-ud-train.conllx",
                    "data/ud/UD_Norwegian-BokmaalNynorsk/no_bokmaalnynorsk-ud-dev.conllx",
                    "data/ud/UD_Norwegian-BokmaalNynorsk/no_bokmaalnynorsk-ud-test.conllx",
                    "data/fastText_data/wiki.no.bokmaalnynorsk.vec.new",
                ),
                'pt': (
                    "data/ud/UD_Portuguese-BosqueGSD/pt_bosquegsd-ud-train.conllx",
                    "data/ud/UD_Portuguese-BosqueGSD/pt_bosquegsd-ud-dev.conllx",
                    "data/ud/UD_Portuguese-BosqueGSD/pt_bosquegsd-ud-test.conllx",
                    "data/fastText_data/wiki.pt.bosquegsd.vec.new",
                ),
                'sv': (
                    "data/ud/UD_Swedish-Talbanken/sv_talbanken-ud-train.conllx",
                    "data/ud/UD_Swedish-Talbanken/sv_talbanken-ud-dev.conllx",
                    "data/ud/UD_Swedish-Talbanken/sv_talbanken-ud-test.conllx",
                    "data/fastText_data/wiki.sv.talbanken.vec.new",
                ),
                'tr': (
                    "data/ud/UD_Turkish-IMST/tr_imst-ud-train.conllx",
                    "data/ud/UD_Turkish-IMST/tr_imst-ud-dev.conllx",
                    "data/ud/UD_Turkish-IMST/tr_imst-ud-test.conllx",
                    "data/fastText_data/wiki.tr.imst.vec.new",
                ),
                'zh': (
                    "data/ud/UD_Chinese-GSD/zh_gsd-ud-train.conllx",
                    "data/ud/UD_Chinese-GSD/zh_gsd-ud-dev.conllx",
                    "data/ud/UD_Chinese-GSD/zh_gsd-ud-test.conllx",
                    "data/fastText_data/wiki.zh.gsd.vec.new",
                )}

        self.args.ftrain = args.ud_dataset[args.lang][0]
        self.args.fdev = args.ud_dataset[args.lang][1]
        self.args.ftest = args.ud_dataset[args.lang][2]
        self.args.fembed = args.ud_dataset[args.lang][3]

        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            logging.info("Preprocess the data")
            
            self.WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)

            tokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.BERT = BertField('bert', pad='[PAD]', bos='[CLS]',
                                    tokenize=tokenizer.encode)

            if args.feat == 'char':
                self.FEAT = CharField('chars', pad=pad, unk=unk, bos=bos,
                                      fix_len=args.fix_len, tokenize=list)
            elif args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert', pad='[PAD]', bos='[CLS]',
                                      tokenize=tokenizer.encode)
            else:
                self.FEAT = Field('tags', bos=bos)
            self.HEAD = Field('heads', bos=bos, use_vocab=False, fn=int)
            self.REL = Field('rels', bos=bos)
            if args.feat in ('char', 'bert'):
                self.fields = CoNLL(FORM=(self.WORD, self.BERT, self.FEAT),
                                    HEAD=self.HEAD, DEPREL=self.REL)
            else:
                self.fields = CoNLL(FORM=(self.WORD, self.BERT), CPOS=self.FEAT,
                                    HEAD=self.HEAD, DEPREL=self.REL)

            train = Corpus.load(args.ftrain, self.fields, args.max_len)
            if args.fembed:
                if args.bert is False:
                    # fasttext
                    embed = Embedding.load(args.fembed, args.lang, unk=args.unk)
                else:
                    embed = None
            else:
                embed = None
            
            self.WORD.build(train, args.min_freq, embed)
            self.FEAT.build(train)
            self.BERT.build(train)
            self.REL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat in ('char', 'bert'):
                self.WORD, self.BERT, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.BERT, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.HEAD, self.REL = self.fields.HEAD, self.fields.DEPREL


        self.puncts = torch.tensor([i for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(args.device)
        self.criterion = nn.CrossEntropyLoss()

        logging.info(f"{self.WORD}\n{self.FEAT}\n{self.BERT}\n{self.HEAD}\n{self.REL}")
        args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_bert': len(self.BERT.vocab),
            'n_rels': len(self.REL.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'bos_index': self.WORD.bos_index
        })
        logging.info(f"n_words {args.n_words} n_feats {args.n_feats} n_bert {args.n_bert} pad_index {args.pad_index} bos_index {args.bos_index}")

    def train(self, loader, self_train=None):
        self.model.train()

        cnt = 0
        for words, bert, feats, arcs, rels in loader:
            if self_train is not None:
                arcs = self_train[cnt]

            self.optimizer.zero_grad()
            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores = self.model(words, bert, feats)
            crf_weight = arc_scores
            arc_scores = self.model.decoder(arc_scores, feats)  # joint_weights
            if self.args.crf:
                if self.args.T_Reg:
                    source_score = self.model.T_Reg(words, bert, feats, self.args.source_model)
                    loss = self.model.crf(crf_weight, arc_scores + self.args.T_beta*source_score, arcs, words, feats)  # crf_weights, joint_weights, heads
                else:
                    loss = self.model.crf(crf_weight, arc_scores, arcs, words, feats)  # crf_weights, joint_weights, heads
            else:
                loss = self.get_loss(arc_scores, arcs, mask)
            if self.args.W_Reg:
                mseloss = self.model.W_Reg()
                loss += mseloss
            if self.args.E_Reg:
                eloss = self.model.E_Reg(words, bert, feats, self.args.source_model, arc_scores)
                loss += eloss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            cnt += 1

    @torch.no_grad()
    def evaluate(self, loader, self_train=None):
        self.model.eval()

        loss, metric = 0, Metric()
        
        cnt = 0
        for words, bert, feats, arcs, rels in loader:
            if self_train is not None:
                arcs = self_train[cnt]

            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores = self.model(words, bert, feats)
            crf_weight = arc_scores
            arc_scores = self.model.decoder(arc_scores, feats)  # joint_weights
            if self.args.crf:
                cur_loss = self.model.crf(crf_weight, arc_scores, arcs, words, feats)  # crf_weights, joint_weights, heads, words, pos
                if self.args.unsupervised:
                    arc_preds = self.model.decode_paskin(arc_scores)
                else:
                    arc_preds = self.model.decode_crf(arc_scores, mask)
                loss += cur_loss
            else:
                loss += self.get_loss(arc_scores, arcs, mask)
                arc_preds = self.model.decode(arc_scores, mask)

            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, arcs, mask)
            cnt += 1

        loss /= len(loader)

        return loss, metric


    @torch.no_grad()
    def get_preds(self, loader):
        self.model.eval()
        loss, metric = 0, Metric()
        arcs_preds = []
            
        for words, bert, feats, arcs, rels in loader:
            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores = self.model(words, bert, feats)
            crf_weight = arc_scores
            arc_scores = self.model.decoder(arc_scores, feats)  # joint_weights
            if self.args.crf:
                cur_loss = self.model.crf(crf_weight, arc_scores, arcs, words, feats)  # crf_weights, joint_weights, heads, words, pos
                if self.args.unsupervised:
                    arc_preds = self.model.decode_paskin(arc_scores)
                else:
                    arc_preds = self.model.decode_crf(arc_scores, mask)
                loss += cur_loss
            else:
                loss += self.get_loss(arc_scores, arcs, mask)
                arc_preds = self.model.decode(arc_scores, mask)
                arcs_preds.append(arc_preds)

        return arcs_preds

    def get_loss(self, arc_scores, arcs, mask):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        arc_loss = self.criterion(arc_scores, arcs)

        return arc_loss