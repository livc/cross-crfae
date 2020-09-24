import torch
from parser.modules import crfae


class HardEM():
    def __init__(self, model, args, laplace_smoothing_k=1):
        self.model = model
        self.k = laplace_smoothing_k  # laplace-smoothing-k
        self.args = args
        self.device = args.device
        self.unsupervised = args.unsupervised

    @torch.no_grad()
    def step(self, loader, train_arcs_preds=None):
        self.model.eval()

        count = torch.zeros_like(self.model.multinomial.data).fill_(self.k)
        cnt = 0
        for words, bert, feats, arcs, _ in loader:
            if train_arcs_preds is not None:
                arcs = train_arcs_preds[cnt]

            mask = words.ne(self.args.pad_index)
            mask[:, 0] = 0
            self.batch_size, self.seq_len = words.shape
            if self.unsupervised:
                arc_scores = self.model(words, bert, feats)
                crf_weight = arc_scores
                if self.args.decoder:
                    arc_scores = self.model.decoder(arc_scores, feats)  # joint_weights
                if self.args.crf:
                    trees = self.model.decode_paskin(arc_scores)
            else:
                trees = arcs
            cnt += 1

            batch_size, _ = trees.size()

            sent_idx = torch.arange(batch_size, device=self.device).contiguous().view(-1, 1).long()

            heads_token_id = feats.data[sent_idx, trees[:, 1:]].contiguous().view(-1)
            children_token_id = feats.data[:, 1:].contiguous().view(-1)

            m, n = count.size()
            linear_idx = heads_token_id * m + children_token_id

            one_tensor = torch.tensor([1], dtype=torch.float, device=self.device).expand_as(linear_idx)
            count.put_(linear_idx, one_tensor, accumulate=True)

        count /= (count.sum(dim=1).contiguous().view(-1, 1))
        self.model.multinomial.data = count        
