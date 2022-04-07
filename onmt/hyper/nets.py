import torch as pt
import torch.nn as nn
import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath
from onmt.hyper.hypeop import *
from onmt.utils.misc import aeq, sequence_mask
from onmt.modules.sparse_activations import sparsemax

gt.PoincareBall.proju()
class MobiusLinear(nn.Module):

    def __init__(self, in_features, out_features, nonlin=None, c=1.0):
        super(MobiusLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        k = (1 / in_features) ** 0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(in_features, out_features).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(out_features).zero_())

    def forward(self, inputs):
        hyp_b = exp_map_zero(self.b)

        wx = mob_mat_mul(self.w, inputs)

        return mob_add(wx, hyp_b)


class MobiusGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, c=1.0, bias=True):
        super(MobiusGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = (1 / hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.u_z = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_r = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_h = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.b_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.b_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.b_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.layernorm = nn.ModuleList(
            [
                nn.ModuleList([nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)])
                for _ in range(2)
            ]
        )

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = mob_mat_mul(W, h)
        U_otimes_x = mob_mat_mul(U, x)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)

        return mob_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden, neg_direc=False):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(self.layernorm[0][0](log_map_zero(z)) if neg_direc else self.layernorm[1][0](log_map_zero(z)))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(self.layernorm[0][1](log_map_zero(r)) if neg_direc else self.layernorm[1][1](log_map_zero(r)))

        r_point_h = mob_pointwise_prod(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)
        # h_tilde = th.tanh(log_map_zero(h_tilde)) # non-linearity

        minus_h_oplus_htilde = mob_add(-hidden, h_tilde)
        new_h = mob_add(hidden, mob_pointwise_prod(minus_h_oplus_htilde, z))

        return new_h


class MobiusGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlin=th.tanh, hyperbolic_input=True, hyperbolic_hidden_state0=True, c=1.0, bidirectional=False, default_dtype=th.float64):
        super(MobiusGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_dtype = default_dtype

        self.gru_cell_p = MobiusGRUCell(input_size, hidden_size)
        self.gru_cell_n = MobiusGRUCell(input_size, hidden_size)

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros((2, batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device)

    def forward(self, input, h0=None):
        # hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        # outputs = []
        # for x in inputs.transpose(0, 1):
        #     hidden = self.gru_cell(x, hidden)
        #     outputs += [hidden]
        # return th.stack(outputs).transpose(0, 1)
        is_packed = isinstance(input, th.nn.utils.rnn.PackedSequence)
        num_directions = 2
        # if is_packed:
        input, batch_sizes = input[:2]
        max_batch_size = int(batch_sizes[0])
        # else:
        #     batch_sizes = None
        #     max_batch_size = input.size(1)
        if h0 is None:
            # h0 = input.new_zeros(
            #     self.num_layers * num_directions, max_batch_size, self.hidden_size, requires_grad=False
            # )
            h0 = self.init_gru_state(max_batch_size, self.hidden_size, input.device)
        h0 = h0.unbind(0)
        # if self.bias is not None:
        #     biases = self.bias
        # else:
        #     biases = (None,) * self.num_layers
        # out = input
        num_steps = batch_sizes.size(0)
        last_batch_size_n = batch_sizes[num_steps - 1]
        last_batch_size_p = batch_sizes[0]
        input_offset_n = input.shape[0]
        input_offset_p = 0
        h_last_p = []
        input_hx_n = h0[1]
        hx_n = input_hx_n[0:batch_sizes[num_steps - 1], :]
        hx_p = h0[0]
        outs_p = []
        outs_n = []
        for i in range(num_steps):
            p_idx = i
            n_idx = num_steps - 1 - p_idx

            batch_size_n = batch_sizes[n_idx]
            batch_size_p = batch_sizes[p_idx]

            step_input_p = input[input_offset_p:input_offset_p + batch_size_p, :]
            step_input_n = input[input_offset_n - batch_size_n:input_offset_n, :]

            input_offset_n = input_offset_n - batch_size_n
            input_offset_p = input_offset_p + batch_size_p

            inc = batch_size_n - last_batch_size_n
            dec = last_batch_size_p - batch_size_p

            if dec > 0:
                h_last_p.append(hx_p[last_batch_size_p - dec:, :])
                hx_p = hx_p[0:last_batch_size_p - dec, :]
            if inc > 0:
                hx_n = th.cat((hx_n, input_hx_n[last_batch_size_n:batch_size_n, :]), 0)

            last_batch_size_n = batch_size_n
            last_batch_size_p = batch_size_p

            hx_p = self.gru_cell_p(
                hyp_x=step_input_p,
                hidden=hx_p,
                neg_direc=False
                # weight_ih=self.weight_ih[0],
                # weight_hh=self.weight_hh[0],
                # bias=biases[0],
                # nonlin=self.nonlin,
                # c=self.ball.c,
                # layer_norm=self.layernorm[0]
            )
            hx_n = self.gru_cell_n(
                hyp_x=step_input_n,
                hidden=hx_n,
                neg_direc=True
                # weight_ih=self.weight_ih[1],
                # weight_hh=self.weight_hh[1],
                # bias=biases[1],
                # nonlin=self.nonlin,
                # c=self.ball.c,
                # layer_norm=self.layernorm[1]
            )

            outs_p.append(hx_p)
            outs_n.append(hx_n)

        h_last_p.append(hx_p)
        h_last_p.reverse()
        h_last_p = th.cat(h_last_p)
        outs_p = th.cat(outs_p)

        h_last_n = hx_n
        outs_n.reverse()
        outs_n = th.cat(outs_n)

        out = th.cat((outs_n, outs_p), 1)
        out = project_hyp_vec(out)

        if is_packed:
            out = th.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = th.stack([h_last_n, h_last_p])

        return out, ht

class LogSoftmax_hype(nn.Module):

    __constants__ = ['dim']

    def __init__(self, c, dim=None):
        super(LogSoftmax_hype, self).__init__()
        self.dim = dim
        self.ball = gt.PoincareBall(c=c)
        self.layernorm = nn.LayerNorm(50006)
        # self.scale_factor = torch.nn.Parameter(init.constant_(torch.Tensor(1, 50006), 1e4))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        # temp = pmath.logmap0(input, k=-1 / self.ball.c, dim=self.dim)
        return nn.functional.log_softmax(self.layernorm(log_map_zero(input)), self.dim, _stacklevel=5)


class GlobalAttention_hype(nn.Module):

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", c=1.0):
        super(GlobalAttention_hype, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp", "hype_asymm_like", "hype_asymm_exp_exp_dire_neg_var_rec", "hype_asymm_like_rec_neg", "hype_symm", "hype_asymm_like_rec_square","hype_asymm_like_rec", "hype_asymm_exp_exp_dire_neg_var_cons", "hype_asymm_exp_exp_dire_neg_var", "hype_asymm_exp_exp_dire", "hype_symm_bias", "hype_asymm_exp_rec", "hype_asymm_sht", "hype_asymm_ths", "hype_asymm_exp", "hype_asymm_plus", "hype_asymm_exp_rec", "hype_asymm_exp_dire"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        if 'hype' in attn_type:
            self.ball = gt.PoincareBall(c=c)
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func


        # mlp wants it with bias
        self.attn_type == "mlp"
        self.linear_context = MobiusLinear(dim, dim, c=c)
        self.linear_query = MobiusLinear(dim, dim, c=c)
        self.linear_out = MobiusLinear(dim * 2, dim, c=c)
        self.layernorm = nn.LayerNorm(dim)
        self.scorenorm = nn.BatchNorm2d(1)
        # self.beta = MobiusLinear(1, 1, c=c)
        if self.attn_type in ["hype_symm", "hype_asymm_like", "hype_asymm_exp_exp_dire_neg_var_rec", "hype_asymm_exp", "hype_asymm_like_rec_neg", "hype_asymm_like_rec_square","hype_asymm_like_rec", "hype_asymm_exp_exp_dire_neg_var_cons", "hype_asymm_exp_exp_dire_neg_var", "hype_asymm_exp_exp_dire", "hype_asymm_plus", "hype_asymm_exp_rec", "hype_asymm_exp_rec", "hype_symm_bias", "hype_asymm_exp_dire"]:
            self.point = th.nn.Parameter(th.Tensor([100]))
        if self.attn_type == "hype_asymm_plus":
            self.alpha1 = th.nn.Parameter(th.Tensor([0.7]))
            self.alpha2 = th.nn.Parameter(th.Tensor([0.7]))
        if self.attn_type == "hype_symm_bias" or self.attn_type == "hype_asymm_like" or self.attn_type == "hype_asymm_exp_exp_dire_neg_var_rec":
            self.bias = torch.nn.Parameter(torch.tensor([-1.1]))
        if self.attn_type == "hype_asymm_exp_exp_dire" or self.attn_type == "hype_asymm_exp_exp_dire_neg_var" or self.attn_type == "hype_asymm_exp_exp_dire_neg_var_rec":
            self.alpha = th.nn.Parameter(th.Tensor([2.3]))

        if coverage:
            self.linear_cover = MobiusLinear(1, dim, c=c)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        # if self.attn_type in ["general", "dot"]:
        #     if self.attn_type == "general":
        #         h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        #         h_t_ = self.linear_in(h_t_)
        #         h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        #     h_s_ = h_s.transpose(1, 2)
        #     # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        #     return torch.bmm(h_t, h_s_)
        # else:
        dim = self.dim
        # d=[]
        # s=[]
        wq = self.linear_query(h_t)
        wq = wq.unsqueeze(2)
        wq = wq.expand(tgt_batch, tgt_len, src_len, dim)
        uh = self.linear_context(h_s)
        uh = uh.unsqueeze(1)
        uh = uh.expand(src_batch, tgt_len, src_len, dim)
        if "hype_symm" in self.attn_type:
            s = (-1) * poinc_dist(wq, uh).squeeze(-1) * self.point
            if "bias" in self.attn_type:
                s = s + self.bias
            return s
        else:
            kp_norm = wq.norm(p=2, dim=-1)
            so_norm = uh.norm(p=2, dim=-1)
            # if self.attn_type == "hype_asymm_sht":
            #     isa_score = 1 + 1000 * (so_norm - kp_norm)
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * isa_score
            #     return s
            # elif self.attn_type == "hype_asymm_ths":
            #     isa_score = 1 + 1000 * (kp_norm - so_norm)
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * isa_score
            #     return s
            # elif self.attn_type == "hype_asymm_exp":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp(so_norm * (-4)) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_neg":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp(so_norm * (4)) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_rec":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) / th.exp(so_norm * (4)) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_dire":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * so_norm * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_plus":
            #     s = (-1) * (poinc_dist(wq, uh).squeeze(-1) + so_norm * self.alpha1 + kp_norm * self.alpha2) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_exp_dire":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp(so_norm * self.alpha) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_exp_dire_neg_var":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp((-1) * so_norm * self.alpha) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_exp_exp_dire_neg_var_cons":
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp((-4) * so_norm) * self.point
            #     return s
            # elif self.attn_type == "hype_asymm_like_rec":
            #     isa_score = kp_norm - so_norm
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * self.point / isa_score
            #     return s
            # elif self.attn_type == "hype_asymm_like_rec_neg":
            #     isa_score = so_norm - kp_norm
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * self.point / isa_score
            #     return s
            # elif self.attn_type == "hype_asymm_like_rec_square":
            #     isa_score = kp_norm - so_norm
            #     s = (-1) * poinc_dist(wq, uh).squeeze(-1) * self.point / isa_score.pow(2)
            #     return s
            if self.attn_type == "hype_asymm_like":
                isa_score = kp_norm - so_norm
                s = (-1) * poinc_dist(wq, uh).squeeze(-1) * self.point * isa_score.pow(2) + self.bias
                return s
            if self.attn_type == "hype_asymm_exp_exp_dire_neg_var_rec":
                s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp(so_norm * self.alpha) * self.point + self.bias
                return s
        # # isa_score = 1 + 1000 * (so_norm - kp_norm)
        # # isa_score = self.point * (so_norm - kp_norm)
        # # temp = kp_norm - so_norm
        # # isa_score = th.where(temp > 0, th.exp(temp), 1/(temp - 1) + 1)
        # # s = (0.0 - poinc_dist(wq, uh).squeeze(-1)*isa_score)
        # s = (-1) * poinc_dist(wq, uh).squeeze(-1) * isa_score
        # s = (-1) * poinc_dist(wq, uh).squeeze(-1) * th.exp(so_norm * (-4))
        # # s = (-1) * poinc_dist(wq, uh).squeeze(-1) * (so_norm * self.alpha1 + kp_norm * self.alpha2)
        # s = (-1) * poinc_dist(wq, uh).squeeze(-1)
        # return s

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        # @memray: the implementation seems not very correct
        # https://github.com/OpenNMT/OpenNMT-py/issues/867
        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = th.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = nn.functional.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        # align_vectors = align_vectors.view(batch, target_l, source_l)
        # align_vectors = align
        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = pmath.weighted_midpoint(memory_bank, batch, target_l, source_l,
                                    align_vectors, k=-1/self.ball.c, reducedim=[1], keepdim=True, dim=-1, lincomb=True, posweight=True
                                    )
        # c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        # c = pmath.sproj(c, k=-1/self.ball.c)
        concat_c = project_hyp_vec(th.cat([c, source], 2).view(batch*target_l, dim*2))
        attn_h = exp_map_zero(th.tanh(self.layernorm(log_map_zero(self.linear_out(concat_c).view(batch, target_l, dim)))))

        if self.attn_type in ["general", "dot"]:
            attn_h = th.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors


class CopyGenerator_hype(nn.Module):

    def __init__(self, input_size, output_size, pad_idx, c=1.0):
        super(CopyGenerator_hype, self).__init__()
        self.linear = MobiusLinear(input_size, output_size, c)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx
        self.layernorm_1 = nn.LayerNorm(output_size)
        self.layernorm_2 = nn.LayerNorm(input_size)
        self.ball = gt.PoincareBall(c=c)
        # self.linear_point = torch.nn.Linear(1, 1)


    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.layernorm_1(log_map_zero(self.linear(hidden)))
        logits[:, self.pad_idx] = -float('inf')
        prob = th.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = th.sigmoid(self.linear_copy(self.layernorm_2(log_map_zero(hidden))))
        # p_copy = torch.sigmoid(self.linear_point(pmath.logmap0(self.linear_copy(hidden), k=-1/self.linear.ball.c)))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = th.mul(prob, 1 - p_copy)
        mul_attn = th.mul(attn, p_copy)
        if mul_attn.dtype == th.float64:
            src_map = src_map.double()
        copy_prob = th.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return th.cat([out_prob, copy_prob], 1)


class MobiusDist2Hyperplane(nn.Module):

    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = gt.PoincareBall(c=c)
        # self.sphere = sphere = geoopt.manifolds.Sphere()
        # self.scale = torch.nn.Parameter(torch.zeros(out_features))
        # point = torch.randn(out_features, in_features) / 4
        # point = pmath.expmap0(point, k=-1/self.ball.c)
        # tangent = torch.randn(out_features, in_features)
        # self.point = geoopt.ManifoldParameter(point, manifold=ball)
        # with torch.no_grad():
        #     self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

        self.z = th.nn.Parameter(th.randn(in_features, out_features)*1e-2)
        self.r = th.nn.Parameter(th.randn(out_features)*1e-2)

    def forward(self, input):
        # input = input.unsqueeze(-2)
        distance = pmath._dist2plane_beta(
            x=input, z=self.z, r=self.r, c=self.ball.c
        )
        return distance

    # def extra_repr(self):
    #     return (
    #         "in_features={in_features}, out_features={out_features}, "
    #         "c={ball.c}".format(
    #             **self.__dict__
    #         )
    #     )
    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={self.ball.c}".format(
                **self.__dict__, self=self
            )
        )