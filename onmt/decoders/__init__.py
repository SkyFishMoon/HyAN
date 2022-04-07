"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder, HyperbolicRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "hyperbolic": HyperbolicRNNDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "HyperbolicRNNDecoder"]
