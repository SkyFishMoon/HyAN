import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.stereographic.math as pmath
import geoopt
from onmt.utils.misc import aeq, sequence_mask
from onmt.modules.sparse_activations import sparsemax
import torch.nn.functional as F
import torch
import torch.nn.init as init
import math

def mobius_linear(
    input,
    weight,
    layer_norm,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    c=torch.tensor(1.0),
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, k=-1/c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=-1/c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=-1/c)
        output = pmath.mobius_add(output, bias, k=-1/c)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, layer_norm(output), k=-1/c)
    output = pmath.project(output, k=-1/c)
    return output


def one_rnn_transform(W, h, U, x, b, c):
    W_otimes_h = pmath.mobius_matvec(W, h, k=-1/c)
    U_otimes_x = pmath.mobius_matvec(U, x, k=-1/c)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, k=-1/c)
    return pmath.mobius_add(Wh_plus_Ux, b, k=-1/c)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    layer_norm,
    nonlin=torch.tanh,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    temp_z_t = pmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, c), k=-1/c)
    temp_r_t = pmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, c), k=-1/c)
    z_t = (layer_norm[0](temp_z_t)).sigmoid()
    r_t = (layer_norm[1](temp_r_t)).sigmoid()

    rh_t = pmath.mobius_pointwise_mul(r_t, hx, k=-1/c)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, c)

    if nonlin is not None:
        h_tilde_log = pmath.logmap0(h_tilde, k=-1/c)
        h_tilde = pmath.expmap0(nonlin(layer_norm[2](h_tilde_log)), k=-1/c)
    # if nonlin is not None:
    #     h_tilde = pmath.mobius_fn_apply(nonlin, layer_norm[2](h_tilde), k=-1/c)

    temp_1 = pmath.mobius_pointwise_mul((1 - z_t), hx, k=-1/c)
    temp_2 = pmath.mobius_pointwise_mul(z_t, h_tilde, k=-1/c)
    h_out = pmath.mobius_add(temp_1, temp_2, k=-1/c)

    # delta_h = pmath.mobius_add(-hx, h_tilde, k=-1/c)
    # h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, k=-1/c), k=-1/c)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    layer_norm,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
    bidirectional=False,
):

    if not hyperbolic_input:
        input = pmath.expmap0(input, k=-1/c)
    outs = []
    if bidirectional:
        if batch_sizes is None:
            if not hyperbolic_hidden_state0:
                hx = pmath.expmap0(h0, k=-1/c)
            else:
                hx = h0
            input_unbinded = input.unbind(0)
            input_unbinded.reverse()
            for t in range(input.size(0)):
                hx = mobius_gru_cell(
                    input=input_unbinded[t],
                    hx=hx,
                    weight_ih=weight_ih,
                    weight_hh=weight_hh,
                    bias=bias,
                    nonlin=nonlin,
                    c=c,
                    layer_norm=layer_norm
                )
                outs.append(hx)
            outs = torch.stack(outs)
            h_last = hx
        else:
            if not hyperbolic_hidden_state0:
                input_hx = pmath.expmap0(h0, k=-1/c)
            else:
                input_hx = h0
            input_offset = input.shape[0]
            num_steps = batch_sizes.size(0)
            last_batch_size = batch_sizes[num_steps - 1]
            hx = input_hx[0:batch_sizes[num_steps - 1], :]
            for i in range(num_steps - 1, -1, -1):
                batch_size = batch_sizes[i]
                inc = batch_size - last_batch_size
                if(inc > 0):
                    hx = torch.cat((hx, input_hx[last_batch_size:batch_size, :]), 0)
                step_input = input[input_offset - batch_size:input_offset, :]
                input_offset = input_offset - batch_size
                last_batch_size = batch_size
                hx = mobius_gru_cell(
                    input=step_input,
                    hx=hx,
                    weight_ih=weight_ih,
                    weight_hh=weight_hh,
                    bias=bias,
                    nonlin=nonlin,
                    c=c,
                    layer_norm=layer_norm
                )
                outs.append(hx)
            outs.reverse()
            outs = torch.cat(outs)
            h_last = hx

    else:
        if not hyperbolic_hidden_state0:
            hx = pmath.expmap0(h0, k=-1/c)
        else:
            hx = h0
        if batch_sizes is None:
            input_unbinded = input.unbind(0)
            for t in range(input.size(0)):
                hx = mobius_gru_cell(
                    input=input_unbinded[t],
                    hx=hx,
                    weight_ih=weight_ih,
                    weight_hh=weight_hh,
                    bias=bias,
                    nonlin=nonlin,
                    c=c,
                    layer_norm=layer_norm
                )
                outs.append(hx)
            outs = torch.stack(outs)
            h_last = hx
        else:
            h_last = []
            # hx = []
            input_offset = 0
            num_steps = batch_sizes.size(0)
            last_batch_size = batch_sizes[0]
            for i in range(num_steps):
                batch_size = batch_sizes[i]
                step_input = input[input_offset:input_offset + batch_size, :]
                input_offset = input_offset + batch_size
                dec = last_batch_size - batch_size
                if (dec > 0):
                    h_last.append(hx[last_batch_size - dec:, :])
                    hx = hx[0:last_batch_size - dec, :]

                last_batch_size = batch_size
                hx = mobius_gru_cell(
                    input=step_input,
                    hx=hx,
                    weight_ih=weight_ih,
                    weight_hh=weight_hh,
                    bias=bias,
                    nonlin=nonlin,
                    c=c,
                    layer_norm=layer_norm
                )
                outs.append(hx)
            h_last.append(hx)
            h_last.reverse()
            h_last = torch.cat(h_last)
            outs = torch.cat(outs)
    return outs, h_last


# class MobiusLinear(torch.nn.Linear):
#     def __init__(
#         self,
#         *args,
#         hyperbolic_input=True,
#         hyperbolic_bias=True,
#         nonlin=None,
#         c=1.0,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.ball = manifold = geoopt.PoincareBall(c=c)
#         if self.bias is not None:
#             if hyperbolic_bias:
#                 self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
#                 with torch.no_grad():
#                     self.bias.set_(pmath.expmap0(init.constant_(self.bias, 0.0), k=-1/self.ball.c))
#         self.weight = geoopt.ManifoldParameter(self.weight, manifold=manifold)
#         with torch.no_grad():
#             self.weight.set_(pmath.expmap0(init.orthogonal_(self.weight), k=-1/self.ball.c))
#         self.hyperbolic_bias = hyperbolic_bias
#         self.hyperbolic_input = hyperbolic_input
#         self.nonlin = nonlin
#
#     def forward(self, input):
#         return mobius_linear(
#             input,
#             weight=self.weight,
#             bias=self.bias,
#             hyperbolic_input=self.hyperbolic_input,
#             nonlin=self.nonlin,
#             hyperbolic_bias=self.hyperbolic_bias,
#             c=self.ball.c,
#         )
#
#     def extra_repr(self):
#         info = super().extra_repr()
#         info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
#         if self.bias is not None:
#             info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
#         return info
#
#
# class MobiusDist2Hyperplane(torch.nn.Module):
#     def __init__(self, in_features, out_features, c=1.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.ball = ball = geoopt.PoincareBall(c=c)
#         self.sphere = sphere = geoopt.manifolds.Sphere()
#         self.scale = torch.nn.Parameter(torch.zeros(out_features))
#         point = torch.randn(out_features, in_features) / 4
#         point = pmath.expmap0(point, k=-1/self.ball.c)
#         tangent = torch.randn(out_features, in_features)
#         self.point = geoopt.ManifoldParameter(point, manifold=ball)
#         with torch.no_grad():
#             self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()
#
#     def forward(self, input):
#         input = input.unsqueeze(-2)
#         distance = pmath.dist2plane(
#             x=input, p=self.point, a=self.tangent, signed=True, k=-1/self.ball.c
#         )
#         return distance * self.scale.exp()
#
#     def extra_repr(self):
#         return (
#             "in_features={in_features}, out_features={out_features}, "
#             "c={self.ball.c}".format(
#                 **self.__dict__, self=self
#             )
#         )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlin=torch.tanh,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
        c=1.0,
        bidirectional=False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.ball = geoopt.PoincareBall(c=c)

        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(
                    torch.Tensor(3 * hidden_size, input_size if (i == 0) or (i == 1) else hidden_size)
                ))
                for i in range(num_layers * num_directions)
            ]
        )
        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers * num_directions)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers * num_directions):
                bias = init.constant_(torch.Tensor(3, hidden_size), 0.0)
                bias = geoopt.ManifoldParameter(
                    pmath.expmap0(bias, k=-1/self.ball.c), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.layernorm = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size),torch.nn.LayerNorm(hidden_size), torch.nn.LayerNorm(hidden_size)])
                for _ in range(num_layers * num_directions)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    # def forward(self, input: torch.Tensor, h0=None):
    #     # input shape: seq_len, batch, input_size
    #     # hx shape: batch, hidden_size
    #     is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
    #     num_directions = 2 if self.bidirectional else 1
    #     if is_packed:
    #         input, batch_sizes = input[:2]
    #         max_batch_size = int(batch_sizes[0])
    #     else:
    #         batch_sizes = None
    #         max_batch_size = input.size(1)
    #     if h0 is None:
    #         h0 = input.new_zeros(
    #             self.num_layers * num_directions, max_batch_size, self.hidden_size, requires_grad=False
    #         )
    #     h0 = h0.unbind(0)
    #     if self.bias is not None:
    #         biases = self.bias
    #     else:
    #         biases = (None,) * self.num_layers
    #     out = input
    #     last_states = []
    #     for i in range(0, self.num_layers, num_directions):
    #         outputs = []
    #         for j in range(num_directions):
    #             if j==1 :
    #                 out_reverse, h_last = mobius_gru_loop(
    #                     input=out,
    #                     h0=h0[i + j],
    #                     weight_ih=self.weight_ih[i + j],
    #                     weight_hh=self.weight_hh[i + j],
    #                     bias=biases[i + j],
    #                     c=self.ball.c,
    #                     layer_norm=self.layernorm[i + j],
    #                     hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
    #                     hyperbolic_input=self.hyperbolic_input or i > 0,
    #                     nonlin=self.nonlin,
    #                     batch_sizes=batch_sizes,
    #                     bidirectional=True
    #                 )
    #                 outputs.append(out_reverse)
    #                 last_states.append(h_last)
    #             else:
    #                 out_front, h_last = mobius_gru_loop(
    #                     input=out,
    #                     h0=h0[i],
    #                     weight_ih=self.weight_ih[i],
    #                     weight_hh=self.weight_hh[i],
    #                     bias=biases[i],
    #                     c=self.ball.c,
    #                     layer_norm=self.layernorm[i],
    #                     hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
    #                     hyperbolic_input=self.hyperbolic_input or i > 0,
    #                     nonlin=self.nonlin,
    #                     batch_sizes=batch_sizes,
    #                 )
    #                 outputs.append(out_front)
    #                 last_states.append(h_last)
    #
    #         out = torch.cat((outputs[0], outputs[1]), 1)
    #         out = pmath.project(out, k=(-1/self.ball.c))
    #     if is_packed:
    #         out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
    #     ht = torch.stack(last_states)
    #     # ht = torch.cat((last_states[0], last_states[1]), 1)
    #     # default api assumes
    #     # out: (seq_len, batch, num_directions * hidden_size)
    #     # ht: (num_layers * num_directions, batch, hidden_size)
    #     # if packed:
    #     # out: (sum(seq_len), num_directions * hidden_size)
    #     # ht: (num_layers * num_directions, batch, hidden_size)
    #     return out, ht
    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        num_directions = 2 if self.bidirectional else 1
        # if is_packed:
        input, batch_sizes = input[:2]
        max_batch_size = int(batch_sizes[0])
        # else:
        #     batch_sizes = None
        #     max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers * num_directions, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        out = input
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
                hx_n = torch.cat((hx_n, input_hx_n[last_batch_size_n:batch_size_n, :]), 0)

            last_batch_size_n = batch_size_n
            last_batch_size_p = batch_size_p

            hx_p = mobius_gru_cell(
                input=step_input_p,
                hx=hx_p,
                weight_ih=self.weight_ih[0],
                weight_hh=self.weight_hh[0],
                bias=biases[0],
                nonlin=self.nonlin,
                c=self.ball.c,
                layer_norm=self.layernorm[0]
            )
            hx_n = mobius_gru_cell(
                input=step_input_n,
                hx=hx_n,
                weight_ih=self.weight_ih[1],
                weight_hh=self.weight_hh[1],
                bias=biases[1],
                nonlin=self.nonlin,
                c=self.ball.c,
                layer_norm=self.layernorm[1]
            )

            outs_p.append(hx_p)
            outs_n.append(hx_n)

        h_last_p.append(hx_p)
        h_last_p.reverse()
        h_last_p = torch.cat(h_last_p)
        outs_p = torch.cat(outs_p)

        h_last_n = hx_n
        outs_n.reverse()
        outs_n = torch.cat(outs_n)

        out = torch.cat((outs_n, outs_p), 1)
        out = pmath.project(out, k=(-1 / self.ball.c))

        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack([h_last_n, h_last_p])

        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)

# def mobius_gru_loop(
#     input: torch.Tensor,
#     h0: torch.Tensor,
#     weight_ih: torch.Tensor,
#     weight_hh: torch.Tensor,
#     bias: torch.Tensor,
#     c: torch.Tensor,
#     layer_norm,
#     # scale_factor: torch.Tensor,
#     batch_sizes=None,
#     hyperbolic_input: bool = False,
#     hyperbolic_hidden_state0: bool = False,
#     nonlin=None
# ):
#     if not hyperbolic_hidden_state0:
#         hx = pmath.expmap0(h0, k=-1/c)
#     else:
#         hx = h0
#     if not hyperbolic_input:
#         input = pmath.expmap0(input, k=-1/c)
#     outs = []
#     if batch_sizes is None:
#         input_unbinded = input.unbind(0)
#         for t in range(input.size(0)):
#             hx = mobius_gru_cell(
#                 input=input_unbinded[t],
#                 hx=hx,
#                 weight_ih=weight_ih,
#                 weight_hh=weight_hh,
#                 bias=bias,
#                 # nonlin=nonlin,
#                 # scale_factor=scale_factor,
#                 c=c,
#                 layer_norm=layer_norm
#             )
#             outs.append(hx)
#         outs = torch.stack(outs)
#         h_last = hx
#     else:
#         h_last = []
#         T = len(batch_sizes) - 1
#         for i, t in enumerate(range(batch_sizes.size(0))):
#             ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
#             hx = mobius_gru_cell(
#                 input=ix,
#                 hx=hx,
#                 weight_ih=weight_ih,
#                 weight_hh=weight_hh,
#                 bias=bias,
#                 # nonlin=nonlin,
#                 # scale_factor=scale_factor,
#                 c=c,
#                 layer_norm=layer_norm
#             )
#             outs.append(hx)
#             if t < T:
#                 hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
#                 h_last.append(ht)
#             else:
#                 h_last.append(hx)
#         h_last.reverse()
#         h_last = torch.cat(h_last)
#         outs = torch.cat(outs)
#     return outs, h_last


class MobiusDist2Hyperplane(torch.nn.Module):

    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = geoopt.PoincareBall(c=c)
        # self.sphere = sphere = geoopt.manifolds.Sphere()
        # self.scale = torch.nn.Parameter(torch.zeros(out_features))
        # point = torch.randn(out_features, in_features) / 4
        # point = pmath.expmap0(point, k=-1/self.ball.c)
        # tangent = torch.randn(out_features, in_features)
        # self.point = geoopt.ManifoldParameter(point, manifold=ball)
        # with torch.no_grad():
        #     self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

        self.z = torch.nn.Parameter(torch.randn(in_features, out_features)*1e-2)
        self.r = torch.nn.Parameter(torch.randn(out_features)*1e-2)

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

    # class MobiusDist2Hyperplane(torch.nn.Module):
#     def __init__(self, in_features, out_features, c=1.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.ball = ball = geoopt.PoincareBall(c=c)
#         self.sphere = sphere = geoopt.manifolds.Sphere()
#         self.scale = torch.nn.Parameter(torch.zeros(out_features))
#         point = torch.randn(out_features, in_features) / 4
#         point = pmath.expmap0(point, k=-1/self.ball.c)
#         tangent = torch.randn(out_features, in_features)
#         self.point = geoopt.ManifoldParameter(point, manifold=ball)
#         with torch.no_grad():
#             self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()
#
#     def forward(self, input):
#         input = input.unsqueeze(-2)
#         distance = pmath.dist2plane(
#             x=input, p=self.point, a=self.tangent, k=-1/self.ball.c, signed=True
#         )
#         return distance * self.scale.exp()

    # def extra_repr(self):
    #     return (
    #         "in_features={in_features}, out_features={out_features}, "
    #         "c={ball.c}".format(
    #             **self.__dict__
    #         )
    #     )
    # def extra_repr(self):
    #     return (
    #         "in_features={in_features}, out_features={out_features}, "
    #         "c={self.ball.c}".format(
    #             **self.__dict__, self=self
    #         )
    #     )


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() / 4, k=-1/self.ball.c))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.layernorm = torch.nn.LayerNorm(self.weight.shape[-1])

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            layer_norm=self.layernorm,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=self.ball.c,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info



# class MobiusGRU(torch.nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         num_layers=1,
#         bias=True,
#         nonlin=None,
#         hyperbolic_input=True,
#         hyperbolic_hidden_state0=True,
#         c=1.0,
#         bidirectional=False,
#
#     ):
#         super().__init__()
#         self.ball = geoopt.PoincareBall(c=c)
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bias = bias
#         self.weight_ih = torch.nn.ParameterList(
#             [
#                 torch.nn.Parameter(
#                     torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
#                 )
#                 for i in range(num_layers)
#             ]
#         )
#         self.weight_hh = torch.nn.ParameterList(
#             [
#                 torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
#                 for _ in range(num_layers)
#             ]
#         )
#         if bias:
#             biases = []
#             for i in range(num_layers):
#                 bias = torch.randn(3, hidden_size) * 1e-5
#                 bias = geoopt.ManifoldParameter(
#                     pmath.expmap0(bias, k=-1/self.ball.c), manifold=self.ball
#                 )
#                 biases.append(bias)
#             self.bias = torch.nn.ParameterList(biases)
#         else:
#             self.register_buffer("bias", None)
#         self.nonlin = nonlin
#         self.hyperbolic_input = hyperbolic_input
#         self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
#         self.reset_parameters()
#         self.layernorm = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size), torch.nn.LayerNorm(hidden_size), torch.nn.LayerNorm(hidden_size)])
#         # self.scale_factor = torch.nn.Parameter(init.constant_(torch.Tensor(1, hidden_size), 1e4))
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
#             torch.nn.init.uniform_(weight, -stdv, stdv)
#
#     def forward(self, input: torch.Tensor, h0=None):
#         # input shape: seq_len, batch, input_size
#         # hx shape: batch, hidden_size
#         is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
#         if is_packed:
#             input, batch_sizes = input[:2]
#             max_batch_size = int(batch_sizes[0])
#         else:
#             batch_sizes = None
#             max_batch_size = input.size(1)
#         if h0 is None:
#             h0 = input.new_zeros(
#                 self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
#             )
#         h0 = h0.unbind(0)
#         if self.bias is not None:
#             biases = self.bias
#         else:
#             biases = (None,) * self.num_layers
#         outputs = []
#         last_states = []
#         out = input
#         for i in range(self.num_layers):
#             out, h_last = mobius_gru_loop(
#                 input=out,
#                 h0=h0[i],
#                 weight_ih=self.weight_ih[i],
#                 weight_hh=self.weight_hh[i],
#                 bias=biases[i],
#                 c=self.ball.c,
#                 hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
#                 hyperbolic_input=self.hyperbolic_input or i > 0,
#                 # nonlin=self.nonlin,
#                 batch_sizes=batch_sizes,
#                 layer_norm=self.layernorm
#             )
#             outputs.append(out)
#             last_states.append(h_last)
#         if is_packed:
#             out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
#         ht = torch.stack(last_states)
#         # default api assumes
#         # out: (seq_len, batch, num_directions * hidden_size)
#         # ht: (num_layers * num_directions, batch, hidden_size)
#         # if packed:
#         # out: (sum(seq_len), num_directions * hidden_size)
#         # ht: (num_layers * num_directions, batch, hidden_size)
#         return out, ht
#     def extra_repr(self):
#         return (
#             "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
#             "hyperbolic_input={hyperbolic_input}, "
#             "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
#             "c={self.ball.c}"
#         ).format(**self.__dict__, self=self, bias=self.bias is not None)


class MobiusGRUCell(torch.nn.Module):

    def __init__(self, input_size, hidden_size, c=1.0, bias=True, num_chunks=3):
        super(MobiusGRUCell, self).__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        # self.scale_factor = torch.nn.Parameter(init.constant_(torch.Tensor(1, hidden_size), 1e4))
        self.layernorm = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_size), torch.nn.LayerNorm(hidden_size), torch.nn.LayerNorm(hidden_size)])
        if bias:
            temp = init.constant_(torch.Tensor(3, hidden_size), 0.0)
            temp = geoopt.ManifoldParameter(
                pmath.expmap0(temp, k=-1/self.ball.c), manifold=self.ball
            )
            self.bias = temp
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        # type: # (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = pmath.expmap0(hx, k=-1/self.ball.c)
        self.check_forward_hidden(input, hx, '')
        return mobius_gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias, self.ball.c, layer_norm=self.layernorm
        )


class GlobalAttention_hype(torch.nn.Module):

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", c=1.0):
        super(GlobalAttention_hype, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp", "hype"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        if attn_type =='hype' :
            self.ball = geoopt.PoincareBall(c=c)
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func


        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.liear_context = MobiusLinear(dim, dim, c=c)
        self.linear_target = MobiusLinear(dim, dim, c=c)
        self.linear_out = MobiusLinear(dim * 2, dim, c=c)
        self.beta = MobiusLinear(1, 1, c=c)

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
        # dim = self.dim
        d=[]
        s=[]
        for i in range(h_t.shape[1]):
            for j in range(h_s.shape[1]):
                d.append((0 - self.beta(pmath.dist(h_t[:, i, :], h_s[:, j, :], k=-1/self.ball.c, keepdim=True, dim=-1))))
            d = torch.t(torch.stack(d).squeeze(-1))
            s.append(d)
        s = torch.stack(s).permute(1,0,2)
            # wq = self.linear_query(h_t.view(-1, dim))
            # wq = wq.view(tgt_batch, tgt_len, 1, dim)
            # wq = wq.expand(tgt_batch, tgt_len, src_len, dim)
            #
            # uh = self.linear_context(h_s.contiguous().view(-1, dim))
            # uh = uh.view(src_batch, 1, src_len, dim)
            # uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            # wquh = torch.tanh(wq + uh)

            # return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
            # return torch.stack(d).t().unsqueeze(1)
        return s
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
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = pmath.weighted_midpoint(memory_bank, batch, target_l, source_l,
                                    align_vectors, k=-1/self.ball.c, reducedim=[2]
                                    )
        # c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

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


class CopyGenerator_hype(torch.nn.Module):
    def __init__(self, input_size, output_size, pad_idx, c=1.0):
        super(CopyGenerator_hype, self).__init__()
        self.linear = MobiusLinear(input_size, output_size, c)
        self.linear_copy = MobiusLinear(input_size, 1, c)
        self.pad_idx = pad_idx

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
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        if mul_attn.dtype == torch.float64:
            src_map = src_map.double()
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)
