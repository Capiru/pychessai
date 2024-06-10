import math

import cairosvg
import chess as ch
import chess.svg as svg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch

from pychessai.constants import ChessBoard, ChessRewards
from pychessai.utils import get_device


def get_players_piece_maps(board: ch.Board):
    # Execution time: 0.000391
    pieces = board.piece_map()
    white_map = dict()
    black_map = dict()
    for k, v in pieces.items():
        if str(v).islower():
            black_map[k] = v
        else:
            white_map[k] = v
    return white_map, black_map


def build_diag_maps():
    diagonal_maps = {
        "A1": [1, 22],
        "A2": [2, 21],
        "A3": [3, 20],
        "A4": [4, 19],
        "A5": [5, 18],
        "A6": [6, 17],
        "A7": [7, 16],
        "A8": [8, 30],
        "B1": [15, 21],
        "B2": [1, 20],
        "B3": [2, 19],
        "B4": [3, 18],
        "B5": [4, 17],
        "B6": [5, 16],
        "B7": [30, 6],
        "B8": [7, 29],
        "C1": [14, 20],
        "C2": [19, 15],
        "C3": [1, 18],
        "C4": [17, 2],
        "C5": [3, 16],
        "C6": [30, 4],
        "C7": [5, 29],
        "C8": [6, 28],
        "D1": [13, 19],
        "D2": [14, 18],
        "D3": [15, 17],
        "D4": [16, 1],
        "D5": [30, 2],
        "D6": [3, 29],
        "D7": [4, 28],
        "D8": [5, 27],
        "E1": [12, 18],
        "E2": [13, 17],
        "E3": [14, 16],
        "E4": [15, 30],
        "E5": [1, 29],
        "E6": [2, 28],
        "E7": [3, 27],
        "E8": [4, 26],
        "F1": [11, 17],
        "F2": [12, 16],
        "F3": [13, 30],
        "F4": [14, 29],
        "F5": [15, 28],
        "F6": [1, 27],
        "F7": [2, 26],
        "F8": [3, 25],
        "G1": [10, 16],
        "G2": [11, 30],
        "G3": [12, 29],
        "G4": [13, 28],
        "G5": [14, 27],
        "G6": [15, 26],
        "G7": [1, 25],
        "G8": [2, 24],
        "H1": [9, 30],
        "H2": [10, 29],
        "H3": [11, 28],
        "H4": [12, 27],
        "H5": [13, 26],
        "H6": [14, 25],
        "H7": [15, 24],
        "H8": [1, 23],
    }

    diag_num_maps = {
        1: "A1B2C3D4E5F6G7H8",
        2: "A2B3C4D5E6F7G8",
        3: "A3B4C5D6E7F8",
        4: "A4B5C6D7E8",
        5: "A5B6C7D8",
        6: "A6B7C8",
        7: "A7B8",
        8: "A8",
        9: "H1",
        10: "H2G1",
        11: "H3G2F1",
        12: "H4G3F2E1",
        13: "H5G4F3E2D1",
        14: "H6G5F4E3D2C1",
        15: "H7G6F5E4D3C2B1",
        16: "A7B6C5D4E3F2G1",
        17: "A6B5C4D3E2F1",
        18: "A5B4C3D2E1",
        19: "A4B3C2D1",
        20: "A3B2C1",
        21: "A2B1",
        22: "A1",
        23: "H8",
        24: "H7G8",
        25: "H6G7F8",
        26: "H5G6F7E8",
        27: "H4G5F6E7D8",
        28: "H3G4F5E6D7C8",
        29: "H2G3F4E5D6C7B8",
        30: "H1G2F3E4D5C6B7A8",
    }
    return diagonal_maps, diag_num_maps


def show_board(board, eval_white, eval_black, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(5, 5))
        plt.ion()
        plt.show()
    svg_img = svg.board(board=board)
    f = open("tmp/board.svg", "w")
    f.write(svg_img)
    f.close()
    cairosvg.svg2png(url="tmp/board.svg", write_to="tmp/image.png")
    img = mpimg.imread("tmp/image.png")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    txt = plt.text(
        0.05,
        0.95,
        f"White: {eval_white:.3f}\nBlack: {eval_black:.3f}",
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    plt.imshow(img)
    plt.pause(0.01)
    txt.set_visible(False)
    return fig


def get_fen_as_tensor(fen):
    pytorch = True
    num_pieces = 6
    num_players = 2
    board_size = 8
    real_planes = 7  # 4 castling 1 black or white
    # attacking_planes = 1
    total_num_planes = num_pieces * num_players + real_planes
    if pytorch:
        input_tensor_size = (total_num_planes, board_size, board_size)
    else:
        input_tensor_size = (board_size, board_size, total_num_planes)
    tensor = torch.zeros(input_tensor_size)
    dic_encoder = {
        "p": 0,
        "P": 1,
        "r": 2,
        "R": 3,
        "n": 4,
        "N": 5,
        "b": 6,
        "B": 7,
        "q": 8,
        "Q": 9,
        "k": 10,
        "K": 11,
    }
    fen_position = fen.split(" ")[0]
    # fen_row_positions = fen_position.split("/")
    row = 0
    file_ = 0
    for char in fen_position:
        if char == "/":
            row += 1
            file_ = 0
        elif str.isnumeric(char):
            file_ += int(char)
        else:
            if pytorch:
                tensor[dic_encoder[char], row, file_] = 1
            else:
                tensor[row, file_, dic_encoder[char]] = 1
            file_ += 1
    return tensor.to(get_device())


def get_board_as_tensor(board):
    player_white = board.turn == ch.WHITE
    num_pieces = ChessBoard.NUM_PIECES
    num_players = ChessBoard.NUM_PLAYERS
    board_size = ChessBoard.BOARD_SIZE
    real_planes = ChessBoard.REAL_PLANES
    total_num_planes = num_pieces * num_players + real_planes

    input_tensor_size = (total_num_planes, board_size, board_size)
    tensor = torch.zeros(input_tensor_size)

    if board.turn == ch.WHITE ^ player_white:
        # If it's opponent's turn
        tensor[12, :, :] = 1
    pieces = [ch.PAWN, ch.KNIGHT, ch.BISHOP, ch.ROOK, ch.QUEEN, ch.KING]
    if player_white:
        colors = [ch.WHITE, ch.BLACK]
        starting_plane = 7
    else:
        colors = [ch.BLACK, ch.WHITE]
        starting_plane = 0
    num_before_draw = math.floor(board.halfmove_clock / 2)
    if num_before_draw >= 64:
        tensor[17, 7, 7] = 1
    tensor[17, num_before_draw % 8, ((num_before_draw) // 8) % 8] = 1
    plane = -1
    player_color = 0
    for color in colors:
        if board.has_kingside_castling_rights(color):
            tensor[13 + player_color * 2, :, :] = 1
        if board.has_queenside_castling_rights(color):
            tensor[14 + player_color * 2, :, :] = 1
        player_color += 1
        for piece in pieces:
            plane += 1
            piece_map = board.pieces(piece, color)
            for pos in piece_map:
                tensor[plane, abs(starting_plane - pos // 8), pos % 8] = 1
    assert plane == int(len(pieces) * 2 - 1)

    return tensor.to(get_device())


def decaying_function_cosdecay(match_len, current_move):
    return math.sin(math.pi * (current_move / match_len) ** 3 - math.pi / 2) / 2 + 0.5


def get_reward_value(winner, player_white, decay=False, current_move=0, match_len=0):
    if winner is None:
        reward = ChessRewards.DRAW
    elif not (winner ^ (player_white)):
        reward = ChessRewards.WIN
    else:
        reward = ChessRewards.LOSE
    if decay:
        reward *= decaying_function_cosdecay(match_len, current_move)
    return reward
