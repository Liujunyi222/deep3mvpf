from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import AttentionalAggregation


class Permute(nn.Module):
    """Permute tensor dimensions inside nn.Sequential."""

    def __init__(self, dims: tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class UTRDataset(Dataset):
    """Dataset for sequence + secondary structure based learning."""

    def __init__(self, sequences, structures, degradation_rates, max_seq_len=None):
        self.sequences = sequences
        self.structures = structures
        self.degradation_rates = degradation_rates
        self.max_seq_len = max_seq_len
        self.nucleotide_dict = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
        self.structure_dict = {".": 0, "(": 1, ")": 2}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        structure = self.structures[idx]
        degradation_rate = self.degradation_rates[idx]

        if self.max_seq_len and len(sequence) > self.max_seq_len:
            sequence = sequence[: self.max_seq_len]
            structure = structure[: self.max_seq_len]

        seq_encoded = self.encode_sequence(sequence)
        debruijn_graph = self.build_debruijn_graph(sequence, k=6)
        structure_graph = self.build_structure_graph(sequence, structure)

        return {
            "sequence": seq_encoded,
            "debruijn_graph": debruijn_graph,
            "structure_graph": structure_graph,
            "degradation_rate": torch.tensor(degradation_rate, dtype=torch.float32),
            "length": len(sequence),
        }

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        encoded = [self.nucleotide_dict.get(nt, 4) for nt in sequence.upper()]
        return torch.tensor(encoded, dtype=torch.long)

    def build_debruijn_graph(self, sequence: str, k: int = 6) -> Data:
        sequence = sequence.upper()
        if len(sequence) < k:
            x = torch.zeros((1, 5 * k), dtype=torch.float32)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

        kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        unique_kmers = list(set(kmers))
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(unique_kmers)}

        edges = []
        for i in range(len(kmers) - 1):
            src = kmer_to_idx[kmers[i]]
            dst = kmer_to_idx[kmers[i + 1]]
            edges.append([src, dst])
        if not edges:
            edges = [[0, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = [self.kmer_to_feature(kmer) for kmer in unique_kmers]
        x = torch.tensor(node_features, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index)

    def kmer_to_feature(self, kmer: str):
        feature = []
        for nt in kmer:
            one_hot = [0] * 5
            idx = self.nucleotide_dict.get(nt, 4)
            one_hot[idx] = 1
            feature.extend(one_hot)
        return feature

    def build_structure_graph(self, sequence: str, structure: str) -> Data:
        seq_len = len(sequence)
        if len(structure) != seq_len:
            if len(structure) > seq_len:
                structure = structure[:seq_len]
            else:
                structure += "." * (seq_len - len(structure))

        pairs = []
        stack = []
        for i, c in enumerate(structure):
            if c == "(":
                stack.append(i)
            elif c == ")" and stack:
                j = stack.pop()
                pairs.append((min(i, j), max(i, j)))

        edges = []
        for i in range(seq_len - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        for i, j in pairs:
            edges.append([i, j])
            edges.append([j, i])
        if not edges:
            edges = [[0, 0], [0, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        node_features = []
        for pos in range(seq_len):
            nt = sequence[pos].upper()
            nuc_onehot = [0] * 5
            nuc_idx = self.nucleotide_dict.get(nt, 4)
            nuc_onehot[nuc_idx] = 1

            struct_c = structure[pos]
            struct_onehot = [0] * 3
            struct_idx = self.structure_dict.get(struct_c, 0)
            struct_onehot[struct_idx] = 1

            rel_pos = pos / (seq_len - 1) if seq_len > 1 else 0.5
            node_features.append(nuc_onehot + struct_onehot + [rel_pos])

        x = torch.tensor(node_features, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index)


class MultiScaleCNN(nn.Module):
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=4)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
                nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2),
                nn.Conv1d(embedding_dim, 128, kernel_size=7, padding=3),
                nn.Conv1d(embedding_dim, 128, kernel_size=9, padding=4),
            ]
        )
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(4)])
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            out = conv(x)
            out = bn(out)
            out = F.relu(out)
            out = self.global_pool(out).squeeze(2)
            conv_outputs.append(out)
        x = torch.cat(conv_outputs, dim=1)
        return self.dropout(x)


class DeBruijnGNN(nn.Module):
    def __init__(self, input_dim: int = 30, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(output_dim, 1))

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.conv3(x, edge_index)
        return self.pool(x, index=batch)


class StructureGNN(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(output_dim, 1))

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.conv3(x, edge_index)
        return self.pool(x, index=batch)


class UTRDegradationPredictor(nn.Module):
    """Multi-branch model for regression and binary classification.

    The optional encoder_type argument is kept for backward compatibility with
    the uploaded training scripts that previously referenced a different model
    implementation.
    """

    def __init__(
        self,
        use_cnn: bool = True,
        use_debruijn: bool = True,
        use_structure: bool = True,
        cnn_multiscale: bool = True,
        encoder_type: str | None = None,
    ):
        super().__init__()

        if encoder_type is not None:
            encoder_type = str(encoder_type).lower()
            if encoder_type in {"cnn", "multiscale", "multi_scale", "default"}:
                cnn_multiscale = True
            elif encoder_type in {"single", "single_scale", "simple_cnn"}:
                cnn_multiscale = False

        self.use_cnn = use_cnn
        self.use_debruijn = use_debruijn
        self.use_structure = use_structure

        self.w_cnn = nn.Parameter(torch.tensor(1.0))
        self.w_deb = nn.Parameter(torch.tensor(1.0))
        self.w_stru = nn.Parameter(torch.tensor(1.0))

        self.cnn_out_dim = 0
        if self.use_cnn:
            if cnn_multiscale:
                self.sequence_cnn = MultiScaleCNN(vocab_size=5, embedding_dim=64)
                self.cnn_out_dim = 512
            else:
                self.sequence_cnn = nn.Sequential(
                    nn.Embedding(5, 64, padding_idx=4),
                    Permute((0, 2, 1)),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                    nn.Flatten(),
                )
                self.cnn_out_dim = 128
            self.sequence_encoder = self.sequence_cnn

        self.deb_out_dim = 0
        if self.use_debruijn:
            self.debruijn_gnn = DeBruijnGNN(input_dim=30, hidden_dim=128, output_dim=256)
            self.deb_out_dim = 256

        self.struct_out_dim = 0
        if self.use_structure:
            self.structure_gnn = StructureGNN(input_dim=9, hidden_dim=128, output_dim=256)
            self.struct_out_dim = 256

        total_dim = self.cnn_out_dim + self.deb_out_dim + self.struct_out_dim
        if total_dim == 0:
            raise ValueError("At least one branch must be enabled.")

        self.layer_norm = nn.LayerNorm(total_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, sequence, debruijn_graph_batch, structure_graph_batch):
        features_list = []
        if self.use_cnn:
            features_list.append(self.sequence_cnn(sequence) * self.w_cnn)
        if self.use_debruijn:
            features_list.append(self.debruijn_gnn(debruijn_graph_batch) * self.w_deb)
        if self.use_structure:
            features_list.append(self.structure_gnn(structure_graph_batch) * self.w_stru)

        combined_features = torch.cat(features_list, dim=1)
        combined_features = self.layer_norm(combined_features)
        return self.fusion_fc(combined_features).squeeze(1)


def collate_fn(batch):
    sequences_list = [item["sequence"] for item in batch]
    sequences_padded = pad_sequence(sequences_list, batch_first=True, padding_value=4)
    debruijn_graphs = Batch.from_data_list([item["debruijn_graph"] for item in batch])
    structure_graphs = Batch.from_data_list([item["structure_graph"] for item in batch])
    degradation_rates = torch.stack([item["degradation_rate"] for item in batch])
    return {
        "sequence": sequences_padded,
        "debruijn_graph": debruijn_graphs,
        "structure_graph": structure_graphs,
        "degradation_rate": degradation_rates,
    }
