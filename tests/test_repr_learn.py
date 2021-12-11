import torch
from mclatte.repr_learn import (
    LstmDecoder,
    LstmEncoder
)

def test_lstm_encoder():
    # Given
    N = 10
    M = 20
    D = 8
    C = 3 
    K = 3
    X = torch.rand(M, N, D)
    A = torch.rand(N, K)
    C_ = torch.rand(N, C)

    # When & Then
    encoder = LstmEncoder(input_dim=D, hidden_dim=C, treatment_dim=K)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    
    optimizer.zero_grad()
    output = encoder(X, A)
    loss = loss_function(output, C_)
    loss.backward()
    optimizer.step()


def test_lstm_decoder():
    # Given
    N = 16
    M = 21
    D = 5
    C = 11 
    X = torch.rand(M, N, D)
    C_ = torch.rand(N, C)

    # When & Then
    decoder = LstmDecoder(hidden_dim=C, output_dim=D, max_seq_len=M)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    optimizer.zero_grad()
    output = decoder(C_)
    loss = loss_function(output, X)
    loss.backward()
    optimizer.step()
