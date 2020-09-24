# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.metric import Metric
# from parser.modules.KM import KmEM

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import logging, os, random


class Train(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=32, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--unk', default='',
                               help='unk token in pretrained embeddings')

        subparser.add_argument('--freeze_feat_emb', action='store_true')
        subparser.add_argument('--freeze_word_emb', action='store_false', help='default not update word emb.')
        subparser.add_argument('--unsupervised', action='store_true')
        subparser.add_argument('--crf', action='store_true')
        subparser.add_argument('--W_Reg', action='store_true')
        subparser.add_argument('--E_Reg', action='store_true')
        subparser.add_argument('--T_Reg', action='store_true')
        subparser.add_argument('--paskin', action='store_true')
        subparser.add_argument('--self_train', action='store_true')
        subparser.add_argument('--max_len', type=int, default=9999)
        subparser.add_argument('--patience', type=int, default=100)
        subparser.add_argument('--epochs', type=int, default=50000)
        subparser.add_argument('--lang', type=str, default='en')
        subparser.add_argument('--load', type=str, default='')
        subparser.add_argument('--W_beta', type=float, default=0.0001)
        subparser.add_argument('--T_beta', type=float, default=0.0001)
        subparser.add_argument('--E_beta', type=float, default=0.0001)
        subparser.add_argument('--output', type=str)
        subparser.add_argument('--lr', type=float, default=1e-3)
        subparser.add_argument('--epsilon', type=float, default=1e-12)
        subparser.add_argument('--mu', type=float, default=0.9)
        subparser.add_argument('--nu', type=float, default=0.9)

        subparser.add_argument('--decay_steps', type=int, default=5000)

        subparser.add_argument('--n_mlp_arc', type=int, default=500)
        subparser.add_argument('--n_lstm_hidden', type=int, default=400)
        subparser.add_argument('--n_lstm_layers', type=int, default=3)
        subparser.add_argument('--n_embed', type=int, default=100)

        # BERT
        subparser.add_argument('--bert', action='store_true')

        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        rrr = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader')
        devices_info = rrr.read().strip().split("\n")
        total, used = devices_info[int(os.environ["CUDA_VISIBLE_DEVICES"])].split(',')
        total = int(total)
        used = int(used)
        max_mem = int(total * random.uniform(0.95, 0.97))
        block_mem = max_mem - used
        x = torch.cuda.FloatTensor(256, 1024, block_mem)
        del x
        rrr.close()

        logging.basicConfig(filename=args.output, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        train_corpus = Corpus.load(args.ftrain, self.fields, args.max_len)
        dev_corpus = Corpus.load(args.fdev, self.fields)
        dev40_corpus = Corpus.load(args.fdev, self.fields, args.max_len)
        test_corpus = Corpus.load(args.ftest, self.fields)
        test40_corpus = Corpus.load(args.ftest, self.fields, args.max_len)

        train = TextDataset(train_corpus, self.fields, args.buckets, crf=args.crf)
        dev = TextDataset(dev_corpus, self.fields, args.buckets, crf=args.crf)
        dev40 = TextDataset(dev40_corpus, self.fields, args.buckets, crf=args.crf)
        test = TextDataset(test_corpus, self.fields, args.buckets, crf=args.crf)
        test40 = TextDataset(test40_corpus, self.fields, args.buckets, crf=args.crf)
        # set the data loaders
        if args.self_train:
            train.loader = batchify(train, args.batch_size)
        else:
            train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        dev40.loader = batchify(dev40, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        test40.loader = batchify(test40, args.batch_size)
        logging.info(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        logging.info(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(dev.buckets)} buckets")
        logging.info(f"{'dev40:':6} {len(dev40):5} sentences, "
              f"{len(dev40.loader):3} batches, "
              f"{len(dev40.buckets)} buckets")
        logging.info(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(test.buckets)} buckets")
        logging.info(f"{'test40:':6} {len(test40):5} sentences, "
              f"{len(test40.loader):3} batches, "
              f"{len(test40.buckets)} buckets")

        logging.info("Create the model")
        self.model = Model(args)
        self.model = self.model.to(args.device)

        if args.E_Reg or args.T_Reg:
            source_model = Model(args)
            source_model = source_model.to(args.device)

        # load model
        if args.load != '':
            logging.info("Load source model")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state = torch.load(args.load, map_location=device)['state_dict']
            state_dict = self.model.state_dict()
            for k, v in state.items():
                if k in ['word_embed.weight']:
                    continue
                state_dict.update({k: v})
            self.model.load_state_dict(state_dict)
            init_params = {}
            for name, param in self.model.named_parameters():
                init_params[name] = param.clone()
            self.model.init_params = init_params

            if args.E_Reg or args.T_Reg:
                state_dict = source_model.state_dict()
                for k, v in state.items():
                    if k in ['word_embed.weight']:
                        continue
                    state_dict.update({k: v})
                source_model.load_state_dict(state_dict)
                init_params = {}
                for name, param in source_model.named_parameters():
                    init_params[name] = param.clone()
                source_model.init_params = init_params

        self.model = self.model.load_pretrained(self.WORD.embed)
        self.model = self.model.to(args.device)

        if args.self_train:
            train_arcs_preds = self.get_preds(train.loader)
            del self.model
            self.model = Model(args)
            self.model = self.model.load_pretrained(self.WORD.embed)
            self.model = self.model.to(args.device)

        if args.E_Reg or args.T_Reg:
            source_model = source_model.load_pretrained(self.WORD.embed)
            source_model = source_model.to(args.device)
            args.source_model = source_model

        self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.epsilon)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))

        # test before train
        if args.load is not '':
            logging.info('\n')

            dev_loss, dev_metric = self.evaluate(dev40.loader)
            test_loss, test_metric = self.evaluate(test40.loader)
            logging.info(f"{'dev40:':4} Loss: {dev_loss:.4f} {dev_metric}")
            logging.info(f"{'test40:':4} Loss: {test_loss:.4f} {test_metric}")

            dev_loss, dev_metric = self.evaluate(dev.loader)
            test_loss, test_metric = self.evaluate(test.loader)
            logging.info(f"{'dev:':4} Loss: {dev_loss:.4f} {dev_metric}")
            logging.info(f"{'test:':4} Loss: {test_loss:.4f} {test_metric}")

        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        logging.info("Begin training")
        if args.unsupervised:
            max_uas = 0.
            cnt = 0
            for epoch in range(1, args.epochs + 1):
                start = datetime.now()

                self.train(train.loader)
                
                logging.info(f"Epoch {epoch} / {args.epochs}:")

                dev_loss, dev_metric = self.evaluate(dev40.loader)
                test_loss, test_metric = self.evaluate(test40.loader)
                logging.info(f"{'dev40:':4} Loss: {dev_loss:.4f} {dev_metric}")
                logging.info(f"{'test40:':4} Loss: {test_loss:.4f} {test_metric}")

                dev_loss, dev_metric = self.evaluate(dev.loader)
                test_loss, test_metric = self.evaluate(test.loader)
                logging.info(f"{'dev:':4} Loss: {dev_loss:.4f} {dev_metric}")
                logging.info(f"{'test:':4} Loss: {test_loss:.4f} {test_metric}")

                t = datetime.now() - start
                logging.info(f"{t}s elapsed\n")
        else:
            for epoch in range(1, args.epochs + 1):
                start = datetime.now()

                if args.self_train:
                    self.train(train.loader, train_arcs_preds)
                else:
                    self.train(train.loader)

                logging.info(f"Epoch {epoch} / {args.epochs}:")
                if args.self_train is False:
                    dev_loss, dev_metric = self.evaluate(dev.loader)
                    logging.info(f"{'dev:':4} Loss: {dev_loss:.4f} {dev_metric}")
                
                t = datetime.now() - start

                # save the model if it is the best so far
                if args.self_train:
                    loss, test_metric = self.evaluate(test.loader)
                    logging.info(f"{'test:':6} Loss: {loss:.4f} {test_metric}")
                else:
                    if dev_metric > best_metric and epoch > args.patience:
                        loss, test_metric = self.evaluate(test.loader)
                        logging.info(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

                        best_e, best_metric = epoch, dev_metric
                        if hasattr(self.model, 'module'):
                            self.model.module.save(args.model)
                        else:
                            self.model.save(args.model)
                        logging.info(f"{t}s elapsed, best epoch {best_e} {best_metric} (saved)\n")
                    else:
                        logging.info(f"{t}s elapsed, best epoch {best_e} {best_metric}\n")
                    total_time += t

                    if epoch - best_e >= args.patience:
                        break
            
            if args.self_train is False:
                self.model = Model.load(args.model)
                logging.info(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
                loss, metric = self.evaluate(test.loader)
                logging.info(f"the score of test at epoch {best_e} is {metric.score:.2%}")
                logging.info(f"average time of each epoch is {total_time / epoch}s, {total_time}s elapsed")