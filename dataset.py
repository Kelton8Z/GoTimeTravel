import os
import torch
from sgfmill import sgf, sgf_moves
import codecs

from config import BOARD_SIZE, SPLIT_RATIO

game_files = os.popen("""find . -type f | grep '.sgf'""").read().split('\n')[:-1]

def training_point(board, move, color):
    board_array = torch.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    for p in board.list_occupied_points():
        board_array[0, p[1][0], p[1][1]] = -1.0 + 2 * int(p[0] == color)
    return board_array, move[0]*BOARD_SIZE+move[1]

afterAIfiles = []
for filePath in game_files:
    if '.sgf' in filePath:
        parts = filePath.split("/")
        if len(parts) > 3:
            fileName = parts[3]
            year = fileName[:4]
            if year.isdigit():
                if int(year) >= 2016:
                    afterAIfiles.append(filePath)

num_games = len(afterAIfiles)
num_training_games = int(num_games * SPLIT_RATIO)
print(len(afterAIfiles))
training_game_files = afterAIfiles[:num_training_games]
training_points = []

testing_game_files = afterAIfiles[num_training_games:]
testing_points = []

def get_training_data(train_or_test):
    if train_or_test == "train":
        game_files = training_game_files
    else:
        game_files = testing_game_files

    for i, game_file in enumerate(game_files):
        print('Processing %s/%s: %s' % (i, len(game_files), game_file))
        num_moves = 0

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

        # add missing ; in the pdg dataset as valid sgf defines properties with (;
        if contents[0] == "(" and contents[1] != ";":
            contents = contents[:1] + ";" + contents[2:]

        game = sgf.Sgf_game.from_string(contents)
        board, plays = sgf_moves.get_setup_and_moves(game)
        for color, move in plays:
            if move is None: continue
            row, col = move
            tp = training_point(board, move, color)
            if train_or_test == "train":
                training_points.append(tp)
            else:
                testing_points.append(tp)
            num_moves += 1

    print(f'Total %s moves: %s', "training" if train_or_test == "train" else "testing", len(training_points) if train_or_test == "train" else len(testing_points))

get_training_data("train")
get_training_data("test")

class GoDataset(torch.utils.data.Dataset):
    def __init__(self, data_points):
        self.data_points = data_points
    def __getitem__(self, index):
        return self.data_points[index][0], self.data_points[index][1]
    def __len__(self):
        return len(self.data_points)

training_dataset = GoDataset(training_points)
test_dataset = GoDataset(testing_points)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)