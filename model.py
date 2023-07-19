import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 19
from config import device

def data_point(board, move, color):
    board_array = torch.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
    for p in board.list_occupied_points():
        board_array[0, p[1][0], p[1][1]] = -1.0 + 2 * int(p[0] == color)
    return board_array, move[0]*BOARD_SIZE+move[1]



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(in_features=32*BOARD_SIZE*BOARD_SIZE, out_features=BOARD_SIZE*BOARD_SIZE)
        self.fc2 = nn.Linear(in_features=BOARD_SIZE*BOARD_SIZE, out_features=BOARD_SIZE*BOARD_SIZE)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*BOARD_SIZE*BOARD_SIZE)
        x = self.dropout(F.relu(self.fc1(x)))      # notice the dropout
        x = self.dropout(self.fc2(x))              # notice the dropout
        x = F.log_softmax(x, dim=1)
        return x

# def preAImodel(board, black_turn):

# def postAImodel(board, black_turn):