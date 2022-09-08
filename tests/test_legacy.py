from match import match
from training import train_value_model
from agents.NegaMaxAgents import NegaMaxAgent
from config import CFG
from training import *
import chess as ch


def training_test(agent):
    CFG.last_index = 0
    CFG.last_policy_index = 0
    CFG.epochs = 100
    CFG.max_patience = 100
    outcome,match_tensor = match(agent,NegaMaxAgent(save_policy=True),save_tensor=True)
    save_tensor(match_tensor)
    print(CFG.memory_batch[1])
    train_dataset = BatchMemoryDataset(CFG.memory_batch[0:3],CFG.last_index)
    train_loader,val_loader = get_data_loader(train_dataset,val_dataset=None,batch_size = 1)
    train_value_model(agent,train_loader)
    print(agent.value_model(match_tensor[0].to(CFG.DEVICE))[0])
    return None

def find_hanging_piece(agent):
    fen = "r3rqk1/2n2pp1/2p2n1p/p2p4/1p1P1Q2/1N1P2NP/PP3PP1/3RR1K1 w - - 6 24"
    board = ch.Board(fen)
    agent.is_white = board.turn
    agent.n_simulations = 300
    move = agent.choose_move(board)
    if ch.Move.from_uci("f4c7") == move[0]:
        return None
    else:
        print(move)
        raise BaseException("Wrong move")

def find_checkmate_in_1(agent):
    fen = "3r2k1/p4ppp/8/1p1PN2P/4Q3/P3KP2/BPr3qP/4R2R b - - 4 23"
    board = ch.Board(fen)
    agent.is_white = board.turn
    agent.n_simulations = 300
    move = agent.choose_move(board)
    if ch.Move.from_uci("g2d2") == move[0]:
        return None
    else:
        print(move)
        raise BaseException("Wrong move")

def find_checkmate_in_2(agent):
    fen = "8/3Q3p/3B1ppk/p1p5/P1Pb4/5P1P/6PK/1q6 w - - 8 40"
    board = ch.Board(fen)
    agent.is_white = board.turn
    agent.n_simulations = 30000
    move = agent.choose_move(board)
    if ch.Move.from_uci("d6f8") == move[0]:
        return None
    else:
        print(move)
        raise BaseException("Wrong move")
    board.push(move[0])
    board.push(ch.Move.from_uci("h6g5"))
    move = agent.choose_move(board)
    if ch.Move.from_uci("d7g4") == move[0]:
        return None
    else:
        print(move)
        raise BaseException("Wrong move")
    agent.n_simulations = 30