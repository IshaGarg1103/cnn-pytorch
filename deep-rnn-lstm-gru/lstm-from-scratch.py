import torch 
from torch import nn
import math
import os
import random
import re
import urllib.request
from torch.nn import functional as F


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
DATA_DIR = "./deep-rnn-lstm-gru/data"
DATA_PATH = os.path.join(DATA_DIR, "timemachine.txt")
PLOT_PATH = "./deep-rnn-lstm-gru/lstm_perplexity.png"
os.makedirs(DATA_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(DATA_DIR, ".matplotlib"))

import matplotlib.pyplot as plt

class Vocab:
    def __init__(self, tokens, min_freq=0):
        counter = {}
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<unk>"] + [
            token for token, freq in self.token_freqs if freq >= min_freq
        ]
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(token) for token in tokens]
        return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])
    def to_tokens(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[int(indices)]

def load_time_machine():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub("[^A-Za-z]+", " ", text).lower()
    tokens = list(text)
    vocab = Vocab(tokens)
    corpus = torch.tensor(vocab[tokens], dtype=torch.long)
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps, device):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)
    def data(pos):
        return corpus[pos: pos + num_steps]
    for i in range(0, len(initial_indices), batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = torch.stack([data(j) for j in batch_indices])
        Y = torch.stack([data(j + 1) for j in batch_indices])
        yield X.to(device), Y.to(device)


def seq_data_iter_sequential(corpus, batch_size, num_steps, device):
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    def data(pos):
        return corpus[pos: pos + num_steps]
    for i in range(0, len(initial_indices), batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = torch.stack([data(j) for j in batch_indices])
        Y = torch.stack([data(j + 1) for j in batch_indices])
        yield X.to(device), Y.to(device)


class LSTMScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma = 0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) *sigma) 
        def triple():
            return (
                init_weight(num_inputs, num_hiddens),
                init_weight(num_hiddens, num_hiddens),
                nn.Parameter(torch.zeros(num_hiddens)),

            )
        self.W_xi, self.W_hi, self.b_i = triple() #input gate.
        self.W_xf, self.W_hf, self.b_f = triple()
        self.W_xo, self.W_ho, self.b_o = triple()
        self.W_xc, self.W_hc, self.b_c = triple()

    def forward(self, inputs, H_C = None):
        if H_C is None:
            H = torch.zeros(inputs.shape[1],
            self.num_hiddens,
            device = inputs.device,
            )
            C = torch.zeros(inputs.shape[1],
            self.num_hiddens,
            device = inputs.device,
            )
        else:
            H,C = H_C
        outputs = []
        for X in inputs:
            I = torch.sigmoid(
                torch.matmul(X, self.W_xi)
                + torch.matmul(H, self.W_hi)
                + self.b_i
            )
            F = torch.sigmoid(
                torch.matmul(X, self.W_xf)
                + torch.matmul(H, self.W_hf)
                + self.b_f
            )
            O = torch.sigmoid(
                torch.matmul(X, self.W_xo)
                + torch.matmul(H, self.W_ho)
                + self.b_o
            )
            C_tilde = torch.tanh(
                torch.matmul(X, self.W_xc)
                + torch.matmul(H, self.W_hc)
                + self.b_c
            )
            C = F*C + I*C_tilde
            H = O*torch.tanh(C)
            outputs.append(H)
        return outputs, (H,C)

class RNNLMScratch(nn.Module):
    def __init__(self, rnn, vocab_size):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.W_hq = nn.Parameter(
            torch.randn(rnn.num_hiddens, vocab_size) * rnn.sigma
        )
        self.b_q = nn.Parameter(torch.zeros(vocab_size))
    def one_hot(self, X):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    def output_layer(self, rnn_outputs):
        outputs = [
            H @ self.W_hq + self.b_q
            for H in rnn_outputs
        ]
        return torch.stack(outputs, dim=1)
    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        return self.output_layer(rnn_outputs), state
    def predict(self, prefix, num_preds, vocab, device):
        state = None
        outputs = [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            y_hat, state = self(X, state)
            if i < len(prefix) - 1:
                outputs.append(vocab[prefix[i + 1]])
            else:
                outputs.append(int(y_hat[:, -1, :].argmax(dim=1)))
        return "".join(vocab.to_tokens(outputs))
def clip_gradients(model, grad_clip_val):
    params = [
        p for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm


def evaluate_perplexity(model, corpus, batch_size, num_steps, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for X, Y in seq_data_iter_sequential(corpus, batch_size, num_steps, device):
            y_hat, _ = model(X)
            loss = loss_fn(
                y_hat.reshape(-1, model.vocab_size),
                Y.reshape(-1),
            )
            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
    model.train()
    return math.exp(total_loss / total_tokens)


def train(
    model, train_corpus, val_corpus, num_epochs, batch_size, num_steps,
    lr, grad_clip_val, device
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_ppls = []
    val_ppls = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for X, Y in seq_data_iter_random(train_corpus, batch_size, num_steps, device):
            optimizer.zero_grad()
            y_hat, _ = model(X)
            loss = loss_fn(
                y_hat.reshape(-1, model.vocab_size),
                Y.reshape(-1),
            )
            loss.backward()
            clip_gradients(model, grad_clip_val)
            optimizer.step()
            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
        train_ppl = math.exp(total_loss / total_tokens)
        val_ppl = evaluate_perplexity(
            model, val_corpus, batch_size, num_steps, loss_fn, device
        )
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1}, "
                f"train perplexity {train_ppl:.2f}, "
                f"val perplexity {val_ppl:.2f}"
            )
    return train_ppls, val_ppls


def plot_perplexity(train_ppls, val_ppls, save_path=PLOT_PATH):
    epochs = range(1, len(train_ppls) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_ppls, label="train_ppl")
    plt.plot(epochs, val_ppls, "--", label="val_ppl")
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved perplexity graph to {save_path}")
    plt.show()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    device = get_device()
    batch_size = 1024
    num_steps = 32
    num_hiddens = 32
    num_epochs = 50
    lr = 4
    grad_clip_val = 1
    corpus, vocab = load_time_machine()
    split = int(len(corpus) * 0.9)
    train_corpus = corpus[:split]
    val_corpus = corpus[split:]
    lstm = LSTMScratch(
        num_inputs=len(vocab),
        num_hiddens=num_hiddens,
    )
    model = RNNLMScratch(
        lstm,
        vocab_size=len(vocab),
    ).to(device)
    print("before training:", model.predict("it has", 20, vocab, device))
    train_ppls, val_ppls = train(
        model,
        train_corpus,
        val_corpus,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_steps=num_steps,
        lr=lr,
        grad_clip_val=grad_clip_val,
        device=device,
    )
    plot_perplexity(train_ppls, val_ppls)
    print("after training:", model.predict("it has", 20, vocab, device))