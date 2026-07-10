"""FastNet model architecture.

FastNet is an encode-process-decode graph neural network for global medium-range weather
prediction (https://journals.ametsoc.org/view/journals/aies/5/3/AIES-D-25-0090.1.xml, with architecture details in
https://arxiv.org/abs/2509.17658). The atmospheric state on the O96 reduced Gaussian grid
is encoded onto a coarser multi-scale icosahedral mesh by a bipartite interaction
network, advanced in time by a stack of mesh-to-mesh interaction networks, and decoded
back onto the grid. The model predicts a per-variable increment that is added to the
input state (residual formulation), advancing the state by one 6-hour step per call;
longer forecasts come from autoregressive rollout.

The static mesh geometry (mesh-node features, plus edge features and edge indices for the
encoder, processor and decoder graphs) is not computed at runtime — it is stored in the
checkpoint as buffers and loaded together with the weights.
"""

import torch
from torch import nn


class BufferList(nn.Module):
    """Hold a list of tensors as buffers named "0", "1", ...

    Unlike a plain list attribute, registered buffers move with the module across devices
    and are saved and loaded through the state dict.
    """

    def __init__(self, tensors: list[torch.Tensor]) -> None:
        super().__init__()
        for i, tensor in enumerate(tensors):
            self.register_buffer(str(i), tensor)

    def __getitem__(self, index: int) -> torch.Tensor:
        return getattr(self, str(index))


class MLP(nn.Module):
    """Two-layer perceptron (Linear-ReLU-Linear), optionally ending in ReLU + LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, final_norm: bool = True) -> None:
        super().__init__()
        # in-place ReLUs are safe here (each acts on a fresh Linear output) and skip a
        # full extra copy of every activation
        modules: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_norm:
            modules += [nn.ReLU(inplace=True), nn.LayerNorm(out_dim)]
        self.layers = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class InteractionNetwork(nn.Module):
    """Bipartite message-passing block (source nodes -> edges -> target nodes).

    For each edge, source-node features are concatenated with edge features and passed
    through ``edge_mlp``; edge messages are sum-aggregated onto their target nodes, and
    ``target_mlp`` maps [target features, aggregated messages] to new target features.
    Targets that receive no edge are left as zeros (non-residual node update).
    """

    def __init__(self, source_dim: int, target_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = MLP(source_dim + edge_dim, hidden_dim, hidden_dim)
        self.target_mlp = MLP(target_dim + hidden_dim, hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        edge_index: torch.Tensor,
        source_features: torch.Tensor,
        edge_features: torch.Tensor,
        target_features: torch.Tensor,
        return_edge_update: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute edge updates and new target-node features.

        Args:
            edge_index: [n_edges, 2] int64 (source id, target id) pairs.
            source_features: [batch, n_source_nodes, source_dim]
            edge_features: [batch, n_edges, edge_dim]
            target_features: [batch, n_target_nodes, target_dim]
            return_edge_update: keep the per-edge messages alive and return them; only
                the processor needs them, for its residual edge-feature update.

        Returns:
            (edge_update [batch, n_edges, hidden] or None, new_target_features
            [batch, n_target_nodes, hidden]); the edge residual is applied by callers.
        """
        # gathering inline (rather than via a local) frees the projected source
        # features as soon as the concatenation is built
        edge_update = self.edge_mlp(
            torch.cat([source_features[:, edge_index[:, 0], :], edge_features], dim=-1)
        )

        # sum-aggregate edge messages onto target nodes
        node_idx = edge_index[:, 1].unsqueeze(1).expand(edge_update.shape)
        aggregated = edge_update.new_zeros(
            edge_update.shape[0], target_features.shape[1], edge_update.shape[2]
        )
        aggregated.scatter_reduce_(-2, node_idx, edge_update, "sum")
        if not return_edge_update:
            edge_update = None

        # update only the (sorted, unique) target nodes that receive edges
        unique_targets = torch.unique(edge_index[:, 1])
        mlp_input = torch.cat(
            [target_features[:, unique_targets], aggregated[:, unique_targets]], dim=-1
        )
        del aggregated
        new_features = self.target_mlp(mlp_input)
        del mlp_input
        out = new_features.new_zeros(
            target_features.shape[0], target_features.shape[1], self.hidden_dim
        )
        index = unique_targets.unsqueeze(0).unsqueeze(2).expand(out.shape[0], -1, out.shape[2])
        out.scatter_(1, index, new_features)
        return edge_update, out


class Encoder(nn.Module):
    """Grid-to-mesh encoder: a single bipartite interaction network."""

    def __init__(self, grid_dim: int, mesh_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.grid2mesh_net = InteractionNetwork(grid_dim, mesh_dim, edge_dim, hidden_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        grid_features: torch.Tensor,
        edge_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        _, mesh_features = self.grid2mesh_net(
            edge_index, grid_features, edge_features, target_features
        )
        return mesh_features


class Processor(nn.Module):
    """Stack of mesh-to-mesh interaction networks with residual edge-feature updates."""

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.interaction_nets = nn.ModuleList(
            InteractionNetwork(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        for net in self.interaction_nets:
            edge_update, node_features = net(
                edge_index, node_features, edge_features, node_features, return_edge_update=True
            )
            edge_features = edge_features + edge_update
        return node_features


class Decoder(nn.Module):
    """Mesh-to-grid decoder: bipartite interaction network + output MLP."""

    def __init__(
        self, mesh_dim: int, grid_dim: int, edge_dim: int, hidden_dim: int, out_dim: int
    ) -> None:
        super().__init__()
        self.interaction_net = InteractionNetwork(mesh_dim, grid_dim, edge_dim, hidden_dim)
        self.out_mlp = MLP(hidden_dim, hidden_dim, out_dim, final_norm=False)

    def forward(
        self,
        edge_index: torch.Tensor,
        mesh_features: torch.Tensor,
        edge_features: torch.Tensor,
        grid_features: torch.Tensor,
    ) -> torch.Tensor:
        _, decoded = self.interaction_net(edge_index, mesh_features, edge_features, grid_features)
        return self.out_mlp(decoded)


class FastNet(nn.Module):
    """FastNet-global weather model.

    Default dimensions match the v1.1 checkpoint (``MetOffice/FastNet-global``): 72
    forecast and 12 non-forecast variables on the 40320-cell O96 grid, with a 10242-node
    icosahedral mesh. The mesh geometry buffers are created empty and only become
    meaningful once a checkpoint has been loaded.
    """

    def __init__(
        self,
        num_forecast_features: int = 72,
        num_nonforecast_features: int = 12,
        mesh_feature_dim: int = 2,
        edge_feature_dim: int = 2,
        hidden_dim: int = 768,
        num_processor_layers: int = 16,
        num_mesh_nodes: int = 10242,
        num_encoder_edges: int = 80640,
        num_mesh_edges: int = 81900,
        num_decoder_edges: int = 120960,
    ) -> None:
        super().__init__()
        grid_dim = num_forecast_features + num_nonforecast_features
        self.num_forecast_features = num_forecast_features

        # static mesh geometry, loaded from the checkpoint
        self.register_buffer("mesh_features", torch.zeros(num_mesh_nodes, mesh_feature_dim))
        self.encoder_edge_features = BufferList([torch.zeros(num_encoder_edges, edge_feature_dim)])
        self.encoder_edge_index = BufferList([torch.zeros(num_encoder_edges, 2, dtype=torch.int64)])
        self.mesh_edge_features = BufferList([torch.zeros(num_mesh_edges, edge_feature_dim)])
        self.mesh_edge_index = BufferList([torch.zeros(num_mesh_edges, 2, dtype=torch.int64)])
        self.decoder_edge_features = BufferList([torch.zeros(num_decoder_edges, edge_feature_dim)])
        self.decoder_edge_index = BufferList([torch.zeros(num_decoder_edges, 2, dtype=torch.int64)])

        self.mesh_edge_embedder = MLP(edge_feature_dim, hidden_dim, hidden_dim)
        self.encoder = Encoder(grid_dim, mesh_feature_dim, edge_feature_dim, hidden_dim)
        self.processor = Processor(num_processor_layers, hidden_dim)
        self.decoder = Decoder(
            hidden_dim, grid_dim, edge_feature_dim, hidden_dim, num_forecast_features
        )

    def forward(
        self, non_forecast_states: torch.Tensor, forecast_state: torch.Tensor
    ) -> torch.Tensor:
        """Advance the atmospheric state by one 6h step.

        Args:
            non_forecast_states: [batch, rollout, grid, num_nonforecast_features] forcings
                and constants; only index 0 of the rollout dimension is used.
            forecast_state: [batch, grid, num_forecast_features] current (normalised) state.

        Returns:
            [batch, 1, grid, num_forecast_features] next state.
        """
        non_forecast_state = non_forecast_states[:, 0]
        grid_features = torch.cat([non_forecast_state, forecast_state], dim=-1)
        batch_size = grid_features.shape[0]

        target_features = self.mesh_features.repeat(batch_size, 1, 1)
        encoder_edge_features = self.encoder_edge_features[0].repeat(batch_size, 1, 1)
        mesh_edge_features = self.mesh_edge_features[0].repeat(batch_size, 1, 1)
        decoder_edge_features = self.decoder_edge_features[0].repeat(batch_size, 1, 1)

        mesh_edge_features = self.mesh_edge_embedder(mesh_edge_features)
        mesh_state = self.encoder(
            self.encoder_edge_index[0], grid_features, encoder_edge_features, target_features
        )
        mesh_state = self.processor(self.mesh_edge_index[0], mesh_state, mesh_edge_features)
        # free the embedded mesh edge features before the decoder

        del mesh_edge_features
        increment = self.decoder(
            self.decoder_edge_index[0], mesh_state, decoder_edge_features, grid_features
        )
        next_state = grid_features[:, :, -self.num_forecast_features :] + increment
        return torch.stack([next_state], dim=1)
