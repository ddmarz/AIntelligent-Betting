import torch
import torch.nn as nn

from dl_model.data import NUM_CONTINUOUS_FEATURES


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.LeakyReLU()

        pooled_len = seq_len // 2
        self.residual_fc = nn.Linear(in_channels * seq_len, hidden_channels * pooled_len)
        self.output_dim = hidden_channels * pooled_len

    def forward(self, x):
        batch_size = x.shape[0]
        residual = self.residual_fc(x.reshape(batch_size, -1))

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.reshape(batch_size, -1)

        return out + residual


class CategoricalTransformerBranch(nn.Module):
    def __init__(self, num_divisions, num_teams, division_dim=8, team_dim=16, num_heads=4, ff_dim=64):
        super().__init__()
        self.division_embedding = nn.Embedding(num_divisions, division_dim)
        self.team_embedding = nn.Embedding(num_teams, team_dim)
        self.division_proj = nn.Linear(division_dim, team_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=team_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_dim = team_dim * 3

    def forward(self, division_idx, home_team_idx, away_team_idx):
        division_tok = self.division_proj(self.division_embedding(division_idx))
        home_tok = self.team_embedding(home_team_idx)
        away_tok = self.team_embedding(away_team_idx)

        tokens = torch.stack([division_tok, home_tok, away_tok], dim=1)
        encoded = self.encoder(tokens)
        return encoded.reshape(encoded.shape[0], -1)


class MultiTaskMatchModel(nn.Module):
    def __init__(self, num_divisions, num_teams, conv_hidden=16, team_dim=16, division_dim=8, fusion_dim=64):
        super().__init__()
        self.conv_branch = ResidualConvBlock(1, conv_hidden, NUM_CONTINUOUS_FEATURES)
        self.categorical_branch = CategoricalTransformerBranch(num_divisions, num_teams, division_dim, team_dim)

        fused_input_dim = self.conv_branch.output_dim + self.categorical_branch.output_dim
        self.fusion = nn.Sequential(nn.Linear(fused_input_dim, fusion_dim), nn.LeakyReLU())

        self.goals_head = nn.Linear(fusion_dim, 2)
        self.shots_head = nn.Linear(fusion_dim, 4)
        self.corners_head = nn.Linear(fusion_dim, 4)

    def forward(self, continuous, division_idx, home_team_idx, away_team_idx):
        conv_out = self.conv_branch(continuous.unsqueeze(1))
        cat_out = self.categorical_branch(division_idx, home_team_idx, away_team_idx)
        fused = self.fusion(torch.cat([conv_out, cat_out], dim=1))

        goals_raw = self.goals_head(fused)
        shots_raw = self.shots_head(fused)
        corners_raw = self.corners_head(fused)

        return {
            "lambda_home_goals": torch.exp(goals_raw[:, 0]),
            "lambda_away_goals": torch.exp(goals_raw[:, 1]),
            "mean_home_shots": torch.exp(shots_raw[:, 0]),
            "mean_away_shots": torch.exp(shots_raw[:, 1]),
            "disp_home_shots": torch.exp(shots_raw[:, 2]),
            "disp_away_shots": torch.exp(shots_raw[:, 3]),
            "mean_home_corners": torch.exp(corners_raw[:, 0]),
            "mean_away_corners": torch.exp(corners_raw[:, 1]),
            "disp_home_corners": torch.exp(corners_raw[:, 2]),
            "disp_away_corners": torch.exp(corners_raw[:, 3]),
        }
