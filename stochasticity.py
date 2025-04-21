import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import csv, ast

class SegmentEncoder(torch.nn.Module):
    """
    encodes sequence segments given a randomly segmented sequence.
    """
    def __init__(self, input_dim=3, emb_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.bilstm = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, segments: list[torch.Tensor]):
        """
        segments expects a list of tensors; the stress pattern of each word to label.
        """
        outputs = []

        for seg in segments:

            emb = self.embedding(seg.unsqueeze(0))  # shape: (1, seq_len, emb_dim)
            out, _ = self.bilstm(emb)

            # Take last hidden state only, since we treat each segment as one token
            fwd = out[0, -1, :out.size(2)//2]
            bwd = out[0, 0, out.size(2)//2:]
            outputs.append(torch.cat([fwd, bwd], dim=-1))  # shape: (2*hidden_dim,)

        return torch.stack(outputs, dim=0)  # (num_segments, seg_hidden_dim)


class POSDecoder(torch.nn.Module):
    """
    labels a sequence of encoded segments.
    """
    def __init__(self, label_emb_dim=32, rnn_hidden=64, num_labels=17):
        super().__init__()

        self.label_emb = torch.nn.Embedding(num_labels + 1, label_emb_dim)  # +1 for START token

        self.gru = torch.nn.GRU(rnn_hidden + label_emb_dim, rnn_hidden, batch_first=True)
        self.out = torch.nn.Linear(rnn_hidden, num_labels)

        self.num_labels = num_labels
        self.START = num_labels  # start of sequence index

    def forward(self, seg_encodings, labels=None):
        # seg_encodings: (batch, token, dimension)
        # labels: (batch, token) if not None
        batch, token, _ = seg_encodings.size()
        outputs = []
        h = None

        prev_labels = torch.full((batch,), self.START, dtype=torch.long, device=seg_encodings.device)

        for t in range(token):
            prev_emb = self.label_emb(prev_labels)  # (batch, label_emb_dim)
            seg_t = seg_encodings[:, t, :]  # (batch, seg_hidden_dim)
            inp = torch.cat([seg_t, prev_emb], dim=-1).unsqueeze(1)  # (batch, 1, D+E)

            out, h = self.gru(inp, h)  # out: (batch, 1, H)
            logits = self.out(out.squeeze(1))  # (batch, num_labels)
            outputs.append(logits)

            if labels is not None:
                prev_labels = labels[:, t]  # teacher forcing
            else:
                prev_labels = torch.argmax(logits, dim=-1)

        return torch.stack(outputs, dim=1)  # (batch, token, num_labels)

    def sample(self, seg_encodings, temperature=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        batch, token, _ = seg_encodings.size()
        outputs = []
        h = None

        prev_labels = torch.full((batch,), self.START, dtype=torch.long, device=seg_encodings.device)

        for t in range(token):
            prev_emb = self.label_emb(prev_labels)
            seg_t = seg_encodings[:, t, :]
            inp = torch.cat([seg_t, prev_emb], dim=-1).unsqueeze(1)

            out, h = self.gru(inp, h)
            logits = self.out(out.squeeze(1)) / temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)
            outputs.append(sampled)
            prev_labels = sampled

        return torch.stack(outputs, dim=1)  # (batch, token)


class Stochasticity(torch.nn.Module):
    """
    an rnn to encode and generate pos labels.
    """
    def __init__(self, rnn_dim, seg_emb_dim=16, lab_emb_dim=32, label_count=17):
        super().__init__()

        if rnn_dim % 2 != 0:
            raise ValueError("rnn_dim must be even")

        self.encoder = SegmentEncoder(3, seg_emb_dim, rnn_dim // 2)
        self.decoder = POSDecoder(lab_emb_dim, rnn_dim, label_count)

    def forward(self, segments: list[list[torch.Tensor]], labels=None, mode='train', temperature=1.0, seed=None):
        # segments: list of List[Tensor], batch of segment lists
        # labels: (batch, token) tensor of gold labels, optional
        seg_batch = [self.encoder(s) for s in segments]  # list of (token, D)
        max_len = max(e.size(0) for e in seg_batch)
        padded = torch.zeros(len(seg_batch), max_len, seg_batch[0].size(-1), device=seg_batch[0].device)
        
        for i, e in enumerate(seg_batch):
            padded[i, :e.size(0)] = e

        if mode == 'train':
            return self.decoder(padded, labels)
        
        elif mode == 'decode':
            return self.decoder.sample(padded, temperature=temperature, seed=None)
        
        else:
            raise ValueError("mode must be either 'train' or 'decode'")


class LabelVocab:
    """maps labels to indices and vice versa. automatically generated."""
    def __init__(self):
        self.label2id = {}
        self.id2label = []

    def add(self, label):
        if label not in self.label2id:
            self.label2id[label] = len(self.id2label)
            self.id2label.append(label)
        return self.label2id[label]

    def encode(self, labels):
        return [self.add(l) for l in labels]

    def __len__(self):
        return len(self.id2label)


class SegmentLabelDataset(Dataset):
    """
    dataset object of segments and labels.
    yields (list[tensors], tensor) representing the sequence of segments and labels.
    """
    def __init__(self, csv_path):
        self.data = []
        self.vocab = LabelVocab()

        skip = True

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:

                # lazy way to skip the first row
                if skip:
                    skip = False
                    continue

                stress_str, duration_str, label_str = row[1], row[2], row[3]

                stress = [int(s) for s in stress_str]

                durations = ast.literal_eval(duration_str)

                labels = label_str.strip().split()
                label_ids = torch.tensor(self.vocab.encode(labels), dtype=torch.long)

                # Split stress pattern into segments
                segments = []
                cursor = 0
                for dur in durations:
                    segment = stress[cursor:cursor + dur]
                    segments.append(torch.tensor(segment, dtype=torch.long))
                    cursor += dur

                self.data.append((segments, label_ids))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)