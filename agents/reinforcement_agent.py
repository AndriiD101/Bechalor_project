import random
import numpy as np
import torch
import torch.nn as nn
from agents.agents_interface import AgentInterface


class Connect4NetLegacy(nn.Module):
    """Legacy architecture — matches checkpoints trained with BatchNorm conv layers."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self._flat_size = self._get_flat_size()
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 6, 7)
            return int(self.conv(dummy).numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self._flat_size = self._get_flat_size()

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, 512),
            nn.LayerNorm(512), # Consistent with FIX 1 in your provided code
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7),
        )
        self._init_weights()

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 6, 7)
            return int(self.conv(dummy).numel())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

class DQNAgent(AgentInterface):
    def __init__(self, player_id: int, model_path: str = None, epsilon: float = 0.0):
        super().__init__(player_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = Connect4Net().to(self.device)
        self.epsilon = epsilon
        
        if model_path:
            self.load(model_path)
        self.policy_net.eval()

    def _board_to_tensor(self, board: np.ndarray, valid_moves: list = None) -> torch.Tensor:
        board_flipped = np.flipud(board)
        my_layer = (board_flipped == self.player_id).astype(np.float32)
        opp_id = 2 if self.player_id == 1 else 1
        opp_layer = (board_flipped == opp_id).astype(np.float32)
        
        valid_layer = np.zeros((6, 7), dtype=np.float32)
        if valid_moves:
            for col in valid_moves:
                valid_layer[:, col] = 1.0
        
        state = np.stack([my_layer, opp_layer, valid_layer], axis=0)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def select_move(self, game) -> int:
        valid_moves = [c for c in range(game.column_count) if game.is_valid_location(game.board, c)]
        if not valid_moves: return -1
        if random.random() < self.epsilon: return random.choice(valid_moves)

        with torch.no_grad():
            state_t = self._board_to_tensor(game.board, valid_moves)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze()

        masked = np.full(7, -1e9)
        for col in valid_moves:
            masked[col] = q_values[col]
        return int(np.argmax(masked))

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        # Auto-detect legacy architecture (BatchNorm keys present in checkpoint)
        if any("running_mean" in k for k in state_dict.keys()):
            print("[INFO] Legacy checkpoint detected (BatchNorm). Loading with Connect4NetLegacy.")
            legacy_net = Connect4NetLegacy().to(self.device)
            legacy_net.load_state_dict(state_dict)
            self.policy_net = legacy_net
        else:
            self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()