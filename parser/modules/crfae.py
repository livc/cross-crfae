import torch
from parser.modules import utils
from parser.modules.definition import LOGZERO

EPS = 1e-12


class CRFAE(torch.autograd.Function):
    def __init__(self, seq_len, batch_size, args=None, is_multi_root=False, max_dependency_len=100, length_constraint_on_root=False):
        super(CRFAE, self).__init__()
        self.args = args
        self.seq_len = seq_len
        if args is None:
            self.device = 'cpu'
        else:
            self.device = args.device
        self.is_multi_root = is_multi_root
        self.max_dependency_len = max_dependency_len
        self.length_constraint_on_root = length_constraint_on_root
        self.batch_size = batch_size

    def forward(self, crf_weights, joint_prior_weights):
        if self.args.device == 'cpu':
            self.crf_weights = crf_weights.type(torch.DoubleTensor)
            self.joint_piror_weights = joint_prior_weights.type(torch.DoubleTensor)
        else:
            self.crf_weights = crf_weights.type(torch.cuda.DoubleTensor)
            self.joint_piror_weights = joint_prior_weights.type(torch.cuda.DoubleTensor)
        self.inside_table, self.log_partition = self.dp_inside_batch(self.crf_weights)
        self.best_score, self.best_tree = self.decoding(self.joint_piror_weights)
        return self.log_partition, self.best_score

    def diff(self):
        outside_table = self.dp_outside_batch(self.inside_table, self.crf_weights)

        (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
         ijss, ikss, kjss, id_span_map, span_id_map) = utils.constituent_indexes(self.seq_len,
                                                                                 self.is_multi_root,
                                                                                 self.max_dependency_len,
                                                                                 self.length_constraint_on_root)

        counts = self.inside_table + outside_table  # shape is batch_size * num_span
        part_count = torch.zeros(self.batch_size, self.seq_len, self.seq_len, dtype=torch.double, device=self.device)
        part_count.fill_(LOGZERO)

        for left_index in range(self.seq_len):
            for right_index in range(left_index + 1, self.seq_len):
                span_id = span_id_map.get((left_index, right_index, utils.get_state_code(0, 1, 1)))
                if span_id is not None:
                    part_count[:, left_index, right_index] = counts[:, span_id]

                span_id = span_id_map.get((left_index, right_index, utils.get_state_code(1, 0, 1)))
                if span_id is not None:
                    part_count[:, right_index, left_index] = counts[:, span_id]

        # alpha = part_count - self.log_partition
        alpha = part_count - self.log_partition.view(self.batch_size, 1, 1)
        diff = torch.exp(alpha)

        return diff

    def backward(self, *grad_output):
        expected_count = self.diff()
        # grad_partition, grad_score, _ = grad_output
        grad_partition, grad_score = grad_output
        grad_partition = grad_partition.contiguous().view(self.batch_size, 1, 1)

        gd_partition_over_w = expected_count * grad_partition

        gd_partition_over_w[:, :, 0].fill_(0.0)
        for i in range(self.seq_len):
            gd_partition_over_w[:, i, i].fill_(0.0)

        grad_score = grad_score.contiguous().view(self.batch_size, 1, 1)

        gd_score_over_joint_w = torch.zeros(self.batch_size, self.seq_len, self.seq_len, dtype=torch.double, device=self.device)
        batch_index = torch.arange(self.batch_size, dtype=torch.long, device=self.device).contiguous().view(-1, 1)  # B * 1
        if self.args.unsupervised:
            head_index = self.best_tree[:, 1:].detach()  # B * (N - 1)
        else:
            head_index = self.heads_token_id
        child_index = torch.arange(1, self.seq_len, dtype=torch.long, device=self.device).contiguous().view(1, -1)  # 1 * (N - 1)

        gd_score_over_joint_w[batch_index, head_index, child_index] = 1.
        gd_score_over_joint_w = gd_score_over_joint_w * grad_score

        return gd_partition_over_w, gd_score_over_joint_w

    def dp_inside_batch(self, weights):
        """

        :param weights:  batch_size * seq_len * seq_len
        :return:
        """
        inside_table = torch.zeros(self.batch_size, self.seq_len*self.seq_len*8, dtype=torch.double, device=self.device)
        inside_table.fill_(LOGZERO)

        m = self.seq_len
        (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
         ijss, ikss, kjss, id_span_map, span_id_map) = utils.constituent_indexes(
            m, self.is_multi_root, self.max_dependency_len, self.length_constraint_on_root
        )

        for ii in seed_spans:
            inside_table[:, ii] = 0.0

        for ii in base_left_spans:
            (l, r, c) = id_span_map[ii]
            inside_table[:, ii] = weights[:, l, r]

        for ii in base_right_spans:
            (l, r, c) = id_span_map[ii]
            inside_table[:, ii] = weights[:, r, l]

        for ij in ijss:
            (l, r, c) = id_span_map[ij]
            if ij in left_spans:
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                prob = inside_table[:, ids] + weights[:, l, r]
                inside_table[:, ij] = utils.logaddexp(inside_table[:, ij], prob)
            elif ij in right_spans:
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                prob = inside_table[:, ids] + weights[:, r, l]
                inside_table[:, ij] = utils.logaddexp(inside_table[:, ij], prob)
            else:
                beta_ik, beta_kj = inside_table[:, ikss[ij]], inside_table[:, kjss[ij]]
                probs = (beta_ik + beta_kj)  # B * k
                probs = utils.logsumexp(probs, axis=1)
                inside_table[:, ij] = utils.logaddexp(inside_table[:, ij], probs)

        id1 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 0)), -1)
        id2 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 1)), -1)

        score1 = inside_table[:, id1]  # B * 1
        score2 = inside_table[:, id2]  # B * 1
        ll = utils.logaddexp(score1, score2)

        return inside_table, ll.contiguous().view(self.batch_size)

    def dp_outside_batch(self, inside_table, weights):
        outside_table = torch.zeros(self.batch_size, self.seq_len*self.seq_len*8, dtype=torch.double, device=self.device)
        outside_table.fill_(LOGZERO)

        m = self.seq_len

        (seed_spans, base_left_spans, base_right_spans, left_spans, right_spans,
         ijss, ikss, kjss, id_span_map, span_id_map) = utils.constituent_indexes(m, self.is_multi_root,
                                                                                   self.max_dependency_len,
                                                                                   self.length_constraint_on_root)

        id1 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 0)), -1)
        id2 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 1)), -1)
        outside_table[:, id1] = 0.0
        outside_table[:, id2] = 0.0

        for ij in reversed(ijss):
            (l, r, c) = id_span_map[ij]
            if ij in left_spans:
                prob = outside_table[:, ij] + weights[:, l, r]
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                outside_table[:, ids] = utils.logaddexp(outside_table[:, ids], prob)
            elif ij in right_spans:
                prob = outside_table[:, ij] + weights[:, r, l]
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                outside_table[:, ids] = utils.logaddexp(outside_table[:, ids], prob)
            else:
                K = len(ikss[ij])
                alpha_ij = outside_table[:, ij].contiguous().view(self.batch_size, 1)  # N * 1
                beta_left = inside_table[:, ikss[ij]]  # N * K

                new_right = alpha_ij + beta_left  # N * K
                outside_table[:, kjss[ij]] = utils.logaddexp(outside_table[:, kjss[ij]], new_right)

                # Problem: The span_id in ikss[ij] may be duplicated.
                for i in range(K):
                    ik = ikss[ij][i]
                    kj = kjss[ij][i]
                    alpha_ij = outside_table[:, ij]
                    beta_right = inside_table[:, kj]
                    new_left = alpha_ij + beta_right
                    outside_table[:, ik] = utils.logaddexp((outside_table[:, ik]), new_left)

        for ij in base_left_spans:
            (l, r, c) = id_span_map[ij]
            prob = outside_table[:, ij] + weights[:, l, r]
            ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 1)), -1)
            outside_table[:, ids] = utils.logaddexp(outside_table[:, ids], prob)

        for ij in base_right_spans:
            (l, r, c) = id_span_map[ij]
            prob = outside_table[:, ij] + weights[:, r, l]
            ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 1)), -1)
            outside_table[:, ids] = utils.logaddexp(outside_table[:, ids], prob)

        return outside_table

    def decoding(self, weights):
        best_score, tree = utils.decoding_batch(weights, self.is_multi_root, self.max_dependency_len, self.length_constraint_on_root, self.device)
        return best_score, tree
