import os
import torch
from sgfmill import sgf, sgf_moves
import codecs

from config import BOARD_SIZE, NUM_TRAINING_GAMES


game_files = os.popen("""find . -type f | grep '.sgf'""").read().split('\n')[:-1]

def training_point(board, move, color):
    board_array = torch.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    for p in board.list_occupied_points():
        board_array[0, p[1][0], p[1][1]] = -1.0 + 2 * int(p[0] == color)
    return board_array, move[0]*BOARD_SIZE+move[1]

afterAIfiles = []
# ALL_PATH = 'pgd/All/'
# NEW_PATH = 'pgd/New/'
# years = []
# for Path in [ALL_PATH, NEW_PATH]:
#     for r, d, f in os.walk(Path):
for filePath in game_files:
    if '.sgf' in filePath:
        parts = filePath.split("/")
        if len(parts) > 3:
            fileName = parts[3]
            year = fileName[:4]
            if year.isdigit():
                if int(year) >= 2016:
                    afterAIfiles.append(filePath)

print(len(afterAIfiles))
training_game_files = afterAIfiles
training_points = []
for i, game_file in enumerate(training_game_files):
    print('Processing %s/%s: %s' % (i, len(training_game_files), game_file))
    num_moves = 0
    # os.popen(game_file)

    try:
        with codecs.open(game_file, 'r', encoding='gb2312') as f:
            contents = f.read()
    except UnicodeDecodeError:
        try:
            with codecs.open(game_file, 'r', encoding='utf-8') as f:
                contents = f.read()
        except:
            with codecs.open(game_file, 'r', encoding='gbk') as f:
                contents = f.read()

    # contents = contents.replace('\n', "")
    # print(contents)
    game = sgf.Sgf_game.from_string(contents)
    board, plays = sgf_moves.get_setup_and_moves(game)
    for color, move in plays:
        if move is None: continue
        row, col = move
        tp = training_point(board, move, color)
        training_points.append(tp)
        board.play(row, col, color)
        num_moves += 1

print('Total training moves: %s' % len(training_points))

class GoDataset(torch.utils.data.Dataset):
    def __init__(self, data_points):
        self.data_points = data_points
    def __getitem__(self, index):
        return self.data_points[index][0], self.data_points[index][1]
    def __len__(self):
        return len(self.data_points)

training_dataset = GoDataset(training_points)
# test_dataset = GoDataset(test_points)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)