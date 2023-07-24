# import tensorflow as tf
import copy
import torch
import pygame
import numpy as np
import itertools
import sys
import networkx as nx
import collections
from pygame import gfxdraw
from model import CNN, data_point
from models.katago import modelconfigs, features
from models.katago.load import load_model
from models.katago.board import Board as KataBoard
from models.katago.model_pytorch import Model
from sgfmill.boards import Board as SgfBoard

from config import BOARD_SIZE, device

# preAImodel = CNN()
# checkpoint = torch.load("CNN_afterAImodel_1epoch.pth")
# preAImodel.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# postAImodel = CNN()
# preAImodel.load_state_dict(checkpoint['model_state_dict'])

model_kind = "b18c384nbt"
# model_kind = "b6c96"
model_config = modelconfigs.config_of_name[model_kind]
preAImodel = Model(model_config, BOARD_SIZE)
postAImodel = Model(model_config, BOARD_SIZE)
ckpt_file = "./models/katago/kata1-b18c384nbt-s7041524736-d3540025399/model.ckpt"
# checkpoint = torch.load("CNN_afterAImodel_1epoch.pth")
# checkpoint = torch.load("./models/katago/kata1-b18c384nbt-s6981484800-d3524616345/model.ckpt")
use_swa = False
ckpt, swa_checkpoint, _ = load_model(
    ckpt_file, use_swa, device
)  # return (model, swa_model, other_state_dict)
preAImodel = ckpt  # .load_state_dict(swa_checkpoint)
postAImodel = ckpt  # .load_state_dict(swa_checkpoint)

# Game constants
BOARD_BROWN = (199, 105, 42)
BOARD_WIDTH = 1000
BOARD_BORDER = 75
STONE_RADIUS = 22
WHITE = (255, 255, 255)

"""
Light Grey: RGB(211, 211, 211)
Silver: RGB(192, 192, 192)
Dark Grey: RGB(169, 169, 169)
Grey: RGB(128, 128, 128)
Dim Grey: RGB(105, 105, 105)
"""
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
TURN_POS = (BOARD_BORDER, 20)
SCORE_POS = (BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER + 30)
DOT_RADIUS = 4


class GameState:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = KataBoard(size=board_size)
        self.moves = []
        self.boards = [self.board.copy()]


gs = GameState(BOARD_SIZE)
rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5,
    "asymPowersOfTwo": 0.0,
}
feats = features.Features(model_config, BOARD_SIZE)

def get_input_feature(gs, rules, feature_idx):
    board = gs.board
    assert(preAImodel.bin_input_shape==[22, 19, 19])
    assert(preAImodel.global_input_shape==[19])
    bin_input_data = np.zeros(shape=[1] + [361, 22], dtype=np.float32)
    global_input_data = np.zeros(
        shape=[1] + preAImodel.global_input_shape, dtype=np.float32
    )
    pla = board.pla
    opp = KataBoard.get_opp(pla)
    move_idx = len(gs.moves)
        
    old_bin = copy.copy(bin_input_data)
    old_global = copy.copy(global_input_data)

    feats.fill_row_features(
        board,
        pla,
        opp,
        gs.boards,
        gs.moves,
        move_idx,
        rules,
        bin_input_data,
        global_input_data,
        idx=0,
    )

    assert(np.any(old_bin!=bin_input_data))
    assert(np.any(old_global!=global_input_data))
    locs_and_values = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            pos = feats.loc_to_tensor_pos(loc, board)
            locs_and_values.append((loc, bin_input_data[0, pos, feature_idx]))
    return locs_and_values, global_input_data


def make_grid(size):
    """Return list of (start_point, end_point pairs) defining gridlines

    Args:
        size (int): size of grid

    Returns:
        Tuple[List[Tuple[float, float]]]: start and end points for gridlines
    """
    start_points, end_points = [], []

    # vertical start points (constant y)
    xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    ys = np.full((size), BOARD_BORDER)
    start_points += list(zip(xs, ys))

    # horizontal start points (constant x)
    xs = np.full((size), BOARD_BORDER)
    ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    start_points += list(zip(xs, ys))

    # vertical end points (constant y)
    xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    ys = np.full((size), BOARD_WIDTH - BOARD_BORDER)
    end_points += list(zip(xs, ys))

    # horizontal end points (constant x)
    xs = np.full((size), BOARD_WIDTH - BOARD_BORDER)
    ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
    end_points += list(zip(xs, ys))

    return (start_points, end_points)


def xy_to_colrow(x, y, size):
    """Convert x,y coordinates to column and row number

    Args:
        x (float): x position
        y (float): y position
        size (int): size of grid

    Returns:
        Tuple[int, int]: column and row numbers of intersection
    """
    inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (size - 1)
    x_dist = x - BOARD_BORDER
    y_dist = y - BOARD_BORDER
    col = int(round(x_dist / inc))
    row = int(round(y_dist / inc))
    return col, row


def colrow_to_xy(col, row, size):
    """Convert column and row numbers to x,y coordinates

    Args:
        col (int): column number (horizontal position)
        row (int): row number (vertical position)
        size (int): size of grid

    Returns:
        Tuple[float, float]: x,y coordinates of intersection
    """
    inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (size - 1)
    x = int(BOARD_BORDER + col * inc)
    y = int(BOARD_BORDER + row * inc)
    return x, y


def has_no_liberties(board, group):
    """Check if a stone group has any liberties on a given board.

    Args:
        board (object): game board (size * size matrix)
        group (List[Tuple[int, int]]): list of (col,row) pairs defining a stone group

    Returns:
        [boolean]: True if group has any liberties, False otherwise
    """
    for x, y in group:
        if x > 0 and board[x - 1, y] == 0:
            return False
        if y > 0 and board[x, y - 1] == 0:
            return False
        if x < board.shape[0] - 1 and board[x + 1, y] == 0:
            return False
        if y < board.shape[0] - 1 and board[x, y + 1] == 0:
            return False
    return True


def get_stone_groups(board, color):
    """Get stone groups of a given color on a given board

    Args:
        board (object): game board (size * size matrix)
        color (str): name of color to get groups for

    Returns:
        List[List[Tuple[int, int]]]: list of list of (col, row) pairs, each defining a group
    """
    size = board.shape[0]
    color_code = 1 if color == "black" else 2
    xs, ys = np.where(board == color_code)
    graph = nx.grid_graph(dim=[size, size])
    stones = set(zip(xs, ys))
    all_spaces = set(itertools.product(range(size), range(size)))
    stones_to_remove = all_spaces - stones
    graph.remove_nodes_from(stones_to_remove)
    return nx.connected_components(graph)


def is_valid_move(col, row, board):
    """Check if placing a stone at (col, row) is valid on board

    Args:
        col (int): column number
        row (int): row number
        board (object): board grid (size * size matrix)

    Returns:
        boolean: True if move is valid, False otherewise
    """
    # TODO: check for ko situation (infinite back and forth)
    if col < 0 or col >= board.shape[0]:
        return False
    if row < 0 or row >= board.shape[0]:
        return False
    return board[col, row] == 0


class Game:
    def __init__(self, size):
        self.board = np.zeros((size, size))
        self.size = size
        self.black_turn = True
        self.prisoners = collections.defaultdict(int)
        self.start_points, self.end_points = make_grid(self.size)

    def init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_WIDTH))
        self.screen = screen
        self.ZOINK = pygame.mixer.Sound("wav/zoink.wav")
        self.CLICK = pygame.mixer.Sound("wav/click.wav")
        self.font = pygame.font.SysFont("arial", 30)

    def clear_screen(self):
        # fill board and add gridlines
        self.screen.fill(BOARD_BROWN)
        for start_point, end_point in zip(self.start_points, self.end_points):
            pygame.draw.line(self.screen, BLACK, start_point, end_point)

        # add guide dots
        guide_dots = [3, self.size // 2, self.size - 4]
        for col, row in itertools.product(guide_dots, guide_dots):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, DOT_RADIUS, BLACK)
            gfxdraw.filled_circle(self.screen, x, y, DOT_RADIUS, BLACK)

        pygame.display.flip()

    def pass_move(self):
        self.black_turn = not self.black_turn
        self.draw()

    def handle_click(self):
        # get board position
        x, y = pygame.mouse.get_pos()
        col, row = xy_to_colrow(x, y, self.size)
        if not is_valid_move(col, row, self.board):
            self.ZOINK.play()
            return

        # update board array
        self.board[col, row] = 1 if self.black_turn else 2

        # get stone groups for black and white
        self_color = "black" if self.black_turn else "white"
        other_color = "white" if self.black_turn else "black"

        # handle captures
        capture_happened = False
        for group in list(get_stone_groups(self.board, other_color)):
            if has_no_liberties(self.board, group):
                capture_happened = True
                for i, j in group:
                    self.board[i, j] = 0
                self.prisoners[self_color] += len(group)

        # handle special case of invalid stone placement
        # this must be done separately because we need to know if capture resulted
        if not capture_happened:
            group = None
            for group in get_stone_groups(self.board, self_color):
                if (col, row) in group:
                    break
            if has_no_liberties(self.board, group):
                self.ZOINK.play()
                self.board[col, row] = 0
                return

        # change turns and draw screen
        self.CLICK.play()
        self.black_turn = not self.black_turn
        self.draw()

    def draw(self):
        # draw stones - filled circle and antialiased ring
        self.clear_screen()
        for col, row in zip(*np.where(self.board == 1)):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, BLACK)
            gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, BLACK)
        for col, row in zip(*np.where(self.board == 2)):
            x, y = colrow_to_xy(col, row, self.size)
            gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, WHITE)
            gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, WHITE)

        sgfmill_board = SgfBoard(19)
        black_points = list(zip(*np.where(self.board == 1)))
        white_points = list(zip(*np.where(self.board == 2)))
        empty_points = list(zip(*np.where(self.board == 0)))
        sgfmill_board.apply_setup(black_points, white_points, empty_points)

        placeholder_move = (-1, -1)
        board, _ = data_point(
            sgfmill_board, placeholder_move, "black" if self.black_turn else "white"
        )

        assert(board.shape==(1,19,19))
        num_bin_input_features = modelconfigs.get_num_bin_input_features(model_config)
        input_spatial = board.unsqueeze(0).repeat(num_bin_input_features, 1, 1, 1) #board[np.newaxis, :, :]  # feature plane, batch, 19, 19
        assert(input_spatial.shape==(22,1,19,19))

        old_spatial = copy.copy(input_spatial)
        
        for feature_idx in range(num_bin_input_features):
            locs_and_values, global_input_feature = get_input_feature(gs, rules, feature_idx) # a list of (loc, bin_input_data[0, pos, feature_idx]) across positions
            for loc, value in locs_and_values:
                input_spatial[feature_idx, 0, KataBoard.loc_x(gs.board, loc), KataBoard.loc_y(gs.board, loc)] = torch.from_numpy(np.array(value))
                
        assert gs.board.board.shape == (421,)
        input_global = torch.from_numpy(global_input_feature)

        # assert(np.any(old_global!=input_global))
        # if gs.moves:
        #     assert(torch.any(old_spatial!=input_spatial))
        
        input_spatial = input_spatial.permute(1, 0, 2, 3)
        output, _ = postAImodel(input_spatial, input_global)

        '''out_value typically represents the model's estimate of the expected outcome of the game from the current position (e.g., the probability of winning).
            out_futurepos is a more advanced feature that might represent some form of prediction about future board positions.
            out_seki, out_ownership, out_scoring, and out_scorebelief_logprobs likely provide various forms of information about the current state of the board or predictions about the final outcome, but aren't directly useful for choosing a move.
        '''
        out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership, out_scoring, out_futurepos, out_seki, out_scorebelief_logprobs = output
        # out_policy (1, num_moves, board_size)
        # First, reshape the tensor to flatten the last two dimensions
        out_policy_flattened = out_policy.view(out_policy.shape[0], -1)

        # Then, use argmax to find the index of the highest score
        prediction = out_policy_flattened.argmax(dim=1)

        # prediction = out_policy.argmax(dim=1, keepdim=True)
        x, y = prediction // BOARD_SIZE, prediction % BOARD_SIZE
        
        print("Prediction: ({}, {})".format(x, y))

        # make a play and update game state
        loc = gs.board.loc(x, y)
        pla = gs.board.pla

        gs.board.play(pla,loc)
        gs.moves.append((pla,loc))
        gs.boards.append(gs.board.copy())

        post_AI_POS = colrow_to_xy(x, y, self.size)

        pointer = self.font.render("a", True, BLACK)
        text_rect = pointer.get_rect()
        post_AI_POS_ADJUSTED = (
            post_AI_POS[0] - text_rect.width // 2,
            post_AI_POS[1] - text_rect.height // 2,
        )
        self.screen.blit(pointer, post_AI_POS_ADJUSTED)

        # x, y = preAImodel(self.board, self.black_turn)
        # pre_AI_POS = colrow_to_xy(x, y, self.size)

        # pointer = self.font.render("b", True, BLACK)
        # text_rect = pointer.get_rect()
        # pre_AI_POS_ADJUSTED = (pre_AI_POS[0] - text_rect.width // 2, pre_AI_POS[1] - text_rect.height // 2)
        # self.screen.blit(pointer, pre_AI_POS_ADJUSTED)

        # text for score and turn info
        score_msg = (
            f"Black's Prisoners: {self.prisoners['black']}"
            + f"     White's Prisoners: {self.prisoners['white']}"
        )
        txt = self.font.render(score_msg, True, BLACK)

        # blit() sends text to the screen
        self.screen.blit(txt, SCORE_POS)
        turn_msg = (
            f"{'Black' if self.black_turn else 'White'} to move. "
            + "Click to place stone, press P to pass."
        )
        txt = self.font.render(turn_msg, True, BLACK)
        self.screen.blit(txt, TURN_POS)

        # flip() updates the screen with new shapes
        pygame.display.flip()

    def update(self):
        # TODO: undo button
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                self.handle_click()
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    self.pass_move()


if __name__ == "__main__":
    g = Game(size=19)
    g.init_pygame()
    g.clear_screen()
    g.draw()

    while True:
        g.update()
        pygame.time.wait(100)
