from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        ###
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        # TODO: model architecture
        
        self.gru = nn.GRU(300, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.lstm = nn.LSTM(300, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        
        if bidirectional:
            self.fcn = nn.Sequential(nn.Linear(hidden_size*2, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_class))
        else:
            self.fcn = nn.Sequential(nn.Linear(hidden_size*1, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, num_class))
        
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        
        # batch = Variable(batch.data, requires_grad=True)
        embedding = self.dropout(self.embed(batch))
        
        output, _ = self.gru(embedding.requires_grad_())
        
        output = output[:, -1, :]
        
        result = self.fcn(output)
        
        return result


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
