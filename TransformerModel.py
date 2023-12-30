import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, verbose = False, device=None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if verbose:
            print("Forward pass now ~~~~~~~~~~~~~~~~")
            print("Shape of input:", src.shape)
            print(src)
        src = self.embedding(src) * math.sqrt(self.d_model)
        if verbose:
            print("Shape after embedding:", src.shape)
            print(src)
        src = self.pos_encoder(src)
        if verbose:
            print("Shape after positional encoding:", src.shape)
            print(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        if verbose:
            print("Shape of mask:", src_mask.shape)
            print(src_mask)
        output = self.transformer_encoder(src, src_mask)
        if verbose:
            print("Shape after transformer encoding:", output.shape)
            print(output)
        output = self.linear(output)
        if verbose:
            print("Shape after linear:", output.shape)
            print(output)
        return output
    
    def __call__(self, src: Tensor, src_mask: Tensor = None, verbose = False) -> Tensor:
        return self.forward(src, src_mask = None, verbose = verbose)
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print("PE: Shape of input to PosEnc.forward():", x.shape)
        # print("PE: Shape of self.pe[:x.size(0)]:", self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)].squeeze(1)
        # print("PE: Shape of output after adding PE:", x.shape)
        # print("PE: Shape of output after dropout:", self.dropout(x).shape)
        return self.dropout(x)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)



def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    print('batch size, seq_len: ', bsz, seq_len)
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data



def get_batch(source: Tensor, i: int, bptt=2048) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target



def evaluate(model: nn.Module, eval_data: Tensor, bptt=2048, device=None, criterion=nn.CrossEntropyLoss(), ntokens=28782) -> float:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            data = data.to(device)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
            # total_loss += criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)



def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))