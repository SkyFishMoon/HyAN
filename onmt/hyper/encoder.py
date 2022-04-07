import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from onmt.hyper.nets import MobiusGRU
from onmt.encoders.encoder import EncoderBase
from onmt.hyper.nets import MobiusLinear


class hyper_gru_encoder(EncoderBase):
    def __init__(self, bidirectional, num_layers,
                 hidden_size, embeddings=None,
                 use_bridge=False, c=1.0):
        super(hyper_gru_encoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False
        self.rnn = MobiusGRU(input_size=embeddings.embedding_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bidirectional=bidirectional,
                             c=c)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(hidden_size,
                                    num_layers,
                                    c)

    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge,
            opt.c)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        emb = self.embeddings(src)
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
            # packed_reverse_emb = pack(reverse_emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        # else:
        #     encoder_final = pmath.mobius_fn_apply(F.relu, encoder_final, k=-1/self.rnn.ball.c)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self,
                           hidden_size,
                           num_layers,
                           c=1.0):

        # LSTM has hidden and cell state, other only one
        number_of_states = 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([MobiusLinear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               c=c)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return result.view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs