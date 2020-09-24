import torch


def Km_expected_count(sent_len, device):
    count = torch.zeros((sent_len, sent_len), device=device)

    for i in range(sent_len):
        if i != 0:
            for j in range(1, sent_len):
                if j != i:
                    count[i, j] += 1. / abs(j - i)

    count[0, 1:] = 1. / (sent_len - 1)  # ROOT
    if sent_len > 2:
        count[1:, 1:] = count[1:, 1:] / count[1:, 1:].sum(dim=0) * (sent_len - 2) / (sent_len - 1)
    return count


class KmEM():
    def __init__(self, model, args, laplace_smoothing_k=1):
        self.model = model
        self.k = laplace_smoothing_k  # laplace-smoothing-k
        self.device = args.device
        self.args = args

    def step(self, loader):
        count = torch.zeros_like(self.model.multinomial.data).fill_(self.k)
        for _, _, feats, _, _ in loader:
            feats = feats.to(self.device)
            batch_size, sent_length = feats.size()
            m, n = count.size()
            km_expected_count = (Km_expected_count(sent_length, self.device)[None, :, :]).repeat(batch_size, 1, 1)
            edge_index = (feats[:, :, None] * m + feats[:, None, :]).view(-1)

            count.put_(edge_index, km_expected_count, accumulate=True)

        count /= (count.sum(dim=1).contiguous().view(-1, 1))
        self.model.multinomial.data = count


class KmInit(torch.autograd.Function):
    pass


if __name__ == '__main__':
    print(Km_expected_count(4))
    print(Km_expected_count(3))
    print(Km_expected_count(2))
