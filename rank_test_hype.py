import numpy as np
from onmt.hyper.nets import MobiusGRU
from onmt.hyper.nets import MobiusGRUCell
from onmt.hyper.nets import MobiusLinear
from onmt.hyper.nets import GlobalAttention_hype
from onmt.hyper.nets import LogSoftmax_hype
import torch
import geoopt
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def batch_gen(batch_size=32, seq_len=10, max_no=100):
    while True:
        x = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)
        y = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)

        X = np.random.randint(5, max_no, size=(batch_size, seq_len - 1))
        start = np.zeros((batch_size, 1), dtype=X.dtype)
        X = np.hstack((start, X))
        Y = np.sort(X, axis=1)

        for ind, batch in enumerate(X):
            for j, elem in enumerate(batch):
                x[ind, j, elem] = 1

        for ind, batch in enumerate(Y):
            for j, elem in enumerate(batch):
                y[ind, j, elem] = 1
        yield x, y


BATCH_SIZE = 64
STEP_SIZE = 10
INPUT_SIZE = 75
CELL_SIZE = 100


class rank_hype(torch.nn.Module):

    def __init__(self, input_size, cell_hize, step_size, bidirectional, c):
        super(rank_hype).__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.encoder = MobiusGRU(input_size=input_size,
                                 hidden_size=cell_hize,
                                 c=c,
                                 bidirectional=bidirectional)
        self.decoder = MobiusGRUCell(
            input_size=input_size,
            hidden_size=CELL_SIZE,
            c=c
        )
        self.attention = GlobalAttention_hype(
            dim=input_size,
            c=c
        )
        self.linear = MobiusLinear(CELL_SIZE, input_size)
        self.gen_func = LogSoftmax_hype
        self.setp_size = step_size
    def forward(self, src, tgt, src_lengths):
        tgt = tgt[:-1]
        lengths_list = src_lengths.view(-1).tolist()
        src = pack(src, lengths_list)
        memory_bank, h_last = self.encoder(src, src_lengths)
        decoder_output = []
        for idx, emb_t in enumerate(tgt.split(1)):
            if idx == 0:
                rnn_output = self.decoder(emb_t, h_last)
            else:
                rnn_output = self.decoder(emb_t, rnn_output)
            decoder_output.append(self.gen_func(self.linear(rnn_output)))



