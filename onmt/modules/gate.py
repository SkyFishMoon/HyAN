""" ContextGate module """
import torch
import torch.nn as nn
import geoopt
from onmt.hyper.nets import MobiusLinear
import geoopt.manifolds.stereographic.math as pmath
from onmt.hyper.nets import MobiusDist2Hyperplane
from onmt.hyper.hypeop import *


def context_gate_factory(gate_type, embeddings_size, decoder_size,
                         attention_size, output_size, c):
    """Returns the correct ContextGate class"""

    gate_types = {'source': SourceContextGate,
                  'target': TargetContextGate,
                  'both': BothContextGate,
                  'hyper': HyperContextGate}

    assert gate_type in gate_types, "Not valid ContextGate type: {0}".format(
        gate_type)
    return gate_types[gate_type](embeddings_size, decoder_size, attention_size,
                                 output_size,c)


class ContextGate(nn.Module):
    """
    Context gate is a decoder module that takes as input the previous word
    embedding, the current decoder state and the attention state, and
    produces a gate.
    The gate can be used to select the input from the target side context
    (decoder state), from the source context (attention state) or both.
    """

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(ContextGate, self).__init__()
        input_size = embeddings_size + decoder_size + attention_size
        self.gate = nn.Linear(input_size, output_size, bias=True)
        self.sig = nn.Sigmoid()
        self.source_proj = nn.Linear(attention_size, output_size)
        self.target_proj = nn.Linear(embeddings_size + decoder_size,
                                     output_size)

    def forward(self, prev_emb, dec_state, attn_state):
        input_tensor = torch.cat((prev_emb, dec_state, attn_state), dim=1)
        z = self.sig(self.gate(input_tensor))
        proj_source = self.source_proj(attn_state)
        proj_target = self.target_proj(
            torch.cat((prev_emb, dec_state), dim=1))
        return z, proj_source, proj_target


class ContextGate_hype(nn.Module):

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, c):
        super(ContextGate_hype, self).__init__()
        input_size = embeddings_size + decoder_size + attention_size
        self.sig = nn.Sigmoid()
        self.gate = MobiusLinear(input_size, output_size, c=c)
        self.gate_layernorm = torch.nn.LayerNorm(output_size)
        self.source_proj = MobiusLinear(attention_size, output_size, c=c)
        self.target_proj = MobiusLinear(embeddings_size + decoder_size,
                                     output_size, c=c)
        self.ball = geoopt.PoincareBall(c=c)


    def forward(self, prev_emb, dec_state, attn_state):
        input_tensor = project_hyp_vec(torch.cat((prev_emb, dec_state, attn_state), dim=1))
        z = self.sig(self.gate_layernorm(log_map_zero(self.gate(input_tensor))))
        proj_source = self.source_proj(attn_state)
        proj_target = self.target_proj(
            project_hyp_vec(torch.cat((prev_emb, dec_state), dim=1)))
        return z, proj_source, proj_target


class SourceContextGate(nn.Module):
    """Apply the context gate only to the source context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(SourceContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(
            prev_emb, dec_state, attn_state)
        return self.tanh(target + z * source)


class TargetContextGate(nn.Module):
    """Apply the context gate only to the target context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(TargetContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(z * target + source)


class BothContextGate(nn.Module):
    """Apply the context gate to both contexts"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(BothContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh((1. - z) * target + z * source)


class HyperContextGate(nn.Module):

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, c=1.0):
        super(HyperContextGate, self).__init__()
        self.context_gate = ContextGate_hype(embeddings_size, decoder_size,
                                        attention_size, output_size, c)
        self.layernorm = torch.nn.LayerNorm(output_size)
        self.ball = geoopt.PoincareBall(c=c)

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        # temp1 = mob_pointwise_prod(target, (1. - z))
        # temp2 = mob_pointwise_prod(source, z)
        # out = log_map_zero(mob_add(temp1, temp2))
        temp = mob_add(-target, source)
        out = log_map_zero(mob_add(target, mob_pointwise_prod(temp, z)))
        out = exp_map_zero(torch.tanh(self.layernorm(out)))
        return out