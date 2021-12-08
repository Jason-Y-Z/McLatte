import torch
from mclatte.config import DEVICE
from mclatte.repr_learn import (
    GRUDDecoder,
    GRUDEncoder,
    LinearDecoder,
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
    X = torch.rand(M, N, D).to(DEVICE)
    A = torch.rand(N, K).to(DEVICE)
    C_ = torch.rand(N, C).to(DEVICE)

    # When & Then
    encoder = LstmEncoder(input_dim=D, hidden_dim=C, treatment_dim=K, bidirectional=False)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    
    optimizer.zero_grad()
    output = encoder(X, A)
    loss = loss_function(output, C_)
    loss.backward()
    optimizer.step()


def test_grud_encoder():
    # Given
    N = 23
    M = 11
    D = 9
    C = 4 
    K = 6
    X = torch.rand(M, N, D).to(DEVICE)
    M_ = torch.randint(0, 1, (M, N, D)).to(DEVICE)
    T = torch.rand(M, N, D).to(DEVICE)
    A = torch.rand(N, K).to(DEVICE)
    C_ = torch.rand(N, C).to(DEVICE)

    # When & Then
    encoder = GRUDEncoder(input_dim=D, hidden_dim=C, treatment_dim=K)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)

    optimizer.zero_grad()
    output = encoder(X, A, T, M_)
    loss = loss_function(output, C_)
    loss.backward()
    optimizer.step()


def test_lstm_decoder():
    # Given
    N = 16
    M = 21
    D = 5
    C = 11 
    X = torch.rand(M, N, D).to(DEVICE)
    C_ = torch.rand(N, C).to(DEVICE)

    # When & Then
    decoder = LstmDecoder(hidden_dim=C, output_dim=D, max_seq_len=M)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    optimizer.zero_grad()
    output = decoder(C_)
    loss = loss_function(output, X)
    loss.backward()
    optimizer.step()


def test_grud_decoder():
    # Given
    N = 18
    M = 14
    D = 3
    C = 5 
    X = torch.rand(M, N, D).to(DEVICE)
    M_ = torch.randint(0, 1, (M, N, C)).to(DEVICE)
    T = torch.rand(M, N, D).to(DEVICE)
    C_ = torch.rand(N, C).to(DEVICE)

    # When & Then
    decoder = GRUDDecoder(hidden_dim=C, output_dim=D, max_seq_len=M)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    optimizer.zero_grad()
    output = decoder(C_, T, M_)
    loss = loss_function(output, X)
    loss.backward()
    optimizer.step()


def test_linear_decoder():
    # Given
    N = 28
    M = 24
    D = 1
    C = 6 
    X = torch.rand(M, N, D).to(DEVICE)
    C_ = torch.rand(N, C).to(DEVICE)

    # When & Then
    decoder = LinearDecoder(hidden_dim=C, output_dim=D, max_seq_len=M)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    optimizer.zero_grad()
    output = decoder(C_)
    loss = loss_function(output, X)
    loss.backward()
    optimizer.step()
