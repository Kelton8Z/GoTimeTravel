import torch
import torch.optim as optim
import torch.nn.functional as F
from models.katago.load import load_model
from models.katago.model_pytorch import Model
from dataset import train_loader, test_loader
from models.katago import modelconfigs, features
from models.katago.board import Board as KataBoard
from models.katago import modelconfigs
from sgfmill.boards import Board as SgfBoard
from config import device, BOARD_SIZE
import copy
import numpy as np
from model import data_point

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
model_kind = "b18c384nbt"
model_config = modelconfigs.config_of_name[model_kind]
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

def predict(board):
    gs.board.board = board
    num_bin_input_features = modelconfigs.get_num_bin_input_features(model_config)
    input_spatial = input_spatial = torch.zeros((num_bin_input_features,1,19,19))  # feature plane, batch, 19, 19
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
    return out_policy_flattened.detach().numpy()[0]
    # Then, use argmax to find the index of the highest score
    prediction = out_policy_flattened.argmax(dim=1)[0]

    k = 10
    top_k_pred_indices = out_policy_flattened.topk(k, dim=1).indices[0]
    assert(prediction==top_k_pred_indices[0])

    # x, y = prediction // BOARD_SIZE, prediction % BOARD_SIZE

    def notLegal(prediction, board):
        if prediction < 0 or prediction >= len(board):
            return True

        if board[prediction] != 0:
            return True
        return False
    
    while notLegal(prediction, board):
        # choose next best move
        top_k_pred_indices = top_k_pred_indices[1:]
        if top_k_pred_indices.shape[0] > 0:
            prediction = top_k_pred_indices[0]
        else:
            # no legal moves, pass
            prediction = -1
            break

    assert(prediction)
    
    return prediction

def train(model, device, train_loader, optimizer, epoch):
    
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # data will be a batch of gs.boards i.e. kata board objects 
        outputs = [torch.tensor(predict(board)) for board in data]
    
        # We then stack the outputs to create a single tensor.
        output = torch.stack(outputs)

        loss = F.nll_loss(output, torch.tensor(target))
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = [torch.tensor(predict(board)) for board in data[:, 0, :, :]]
    
        # We then stack the outputs to create a single tensor.
            output = torch.stack(outputs)
            # output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (float(correct) / len(test_loader.dataset))

model_kind = "b18c384nbt"
model_config = modelconfigs.config_of_name[model_kind]
postAImodel = Model(model_config, BOARD_SIZE)
ckpt_file = "./models/katago/kata1-b18c384nbt-s7041524736-d3540025399/model.ckpt"
# checkpoint = torch.load("./models/katago/kata1-b18c384nbt-s6981484800-d3524616345/model.ckpt")
use_swa = False
ckpt, swa_checkpoint, _ = load_model(
    ckpt_file, use_swa, device
)  # return (model, swa_model, other_state_dict)
preAImodel = ckpt  # .load_state_dict(swa_checkpoint)
postAImodel = ckpt
optimizer = optim.SGD(postAImodel.parameters(), lr=0.01, momentum=0.5)

losses = []
accuracies = []
for epoch in range(0, 10):
    losses.extend(train(postAImodel, device, train_loader, optimizer, epoch))
    accuracies.append(test(postAImodel, device, test_loader))
    print(accuracies[-1])
# losses = [item for sublist in losses for item in sublist]

torch.save({'epoch': 10,
              'model_state_dict': postAImodel.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': losses[-1],
              'acc': accuracies[-1]}, 
        './katago_afterAImodel_checkpoint.pth')