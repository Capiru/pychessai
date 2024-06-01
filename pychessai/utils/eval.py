import chess as ch
import numpy as np

from pychessai.utils.board_utils import build_diag_maps


def get_simple_board_evaluation(board: ch.Board):
    # Execution time: 0.000453
    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0
        elif winner is True:
            return np.inf
        else:
            return -np.inf
    count_black = 0.0
    count_white = 0.0
    fen = board.shredder_fen()
    dic_ = {"p": 1.0, "r": 5.0, "n": 2.5, "b": 3.0, "q": 9.0, "k": 1000.0}
    for char in fen.split(" ")[0]:
        if str.islower(char):
            count_black += dic_[char]
        elif str.isnumeric(char) or char == "/":
            continue
        else:
            try:
                count_white += dic_[char.lower()]
            except Exception as e:
                print(fen, e)
    return count_white - count_black


def handcrafted_eval(board, is_white=True):
    if is_white:
        return get_handcrafted_board_evaluation(board)
    else:
        return -get_handcrafted_board_evaluation(board)


def get_handcrafted_board_evaluation(board: ch.Board):
    # Execution time: 0.000453
    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0
        elif winner is True:
            return 50000
        else:
            return -50000
    eval_black = 0
    eval_white = 0
    fen = board.shredder_fen()
    dic_ = {"p": 100, "r": 500, "n": 325, "b": 340, "q": 900, "k": 10000}
    remainder_dic = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H"}
    white_piece_map = {"p": [], "r": [], "n": [], "b": [], "q": [], "k": []}  # type: ignore
    black_piece_map = {"p": [], "r": [], "n": [], "b": [], "q": [], "k": []}  # type: ignore
    white_piece_position_stringmap = {
        "p": "",
        "r": "",
        "n": "",
        "b": "",
        "q": "",
        "k": "",
    }
    black_piece_position_stringmap = {
        "p": "",
        "r": "",
        "n": "",
        "b": "",
        "q": "",
        "k": "",
    }
    diag_pos_map, diag_num_map = build_diag_maps()
    white_diag_map = {(i + 1): "" for i in range(30)}
    black_diag_map = {(i + 1): "" for i in range(30)}
    board_pos = 0
    for char in fen.split(" ")[0]:
        if str.islower(char):
            board_pos += 1
            eval_black += dic_[char]
            current_pos = remainder_dic[board_pos % 8 + 1] + str(
                (board_pos - 1) // 8 + 1
            )
            black_piece_map[char].append(current_pos)
            black_piece_position_stringmap[char] += current_pos
            diags = diag_pos_map[current_pos]
            black_diag_map[diags[0]] += char
            black_diag_map[diags[1]] += char
        elif str.isnumeric(char):
            board_pos += int(char)
            continue
        elif char == "/":
            continue
        else:
            board_pos += 1
            eval_white += dic_[char.lower()]
            current_pos = remainder_dic[board_pos % 8 + 1] + str(
                (board_pos - 1) // 8 + 1
            )
            white_piece_map[char.lower()].append(current_pos)
            white_piece_position_stringmap[char.lower()] += current_pos
            diags = diag_pos_map[current_pos]
            white_diag_map[diags[0]] += char.lower()
            white_diag_map[diags[1]] += char.lower()
    eval_white += pawn_handcrafted_eval(
        white_piece_map["p"],
        white_piece_position_stringmap,
        black_piece_position_stringmap,
        is_white=True,
    )
    eval_white += bishop_handcrafted_eval(
        white_piece_map["b"], white_diag_map, diag_pos_map
    )
    eval_white += knight_handcrafted_eval(white_piece_map["k"])
    eval_white += rook_handcrafted_eval(
        white_piece_map["r"],
        white_piece_position_stringmap,
        black_piece_position_stringmap,
        is_white=True,
    )
    eval_white += queen_handcrafted_eval(
        white_piece_map["q"],
        black_piece_position_stringmap,
        white_diag_map,
        diag_pos_map,
    )

    eval_black += pawn_handcrafted_eval(
        black_piece_map["p"],
        black_piece_position_stringmap,
        white_piece_position_stringmap,
        is_white=False,
    )
    eval_black += bishop_handcrafted_eval(
        black_piece_map["b"], black_diag_map, diag_pos_map
    )
    eval_black += knight_handcrafted_eval(black_piece_map["k"])
    eval_black += rook_handcrafted_eval(
        black_piece_map["r"],
        black_piece_position_stringmap,
        white_piece_position_stringmap,
        is_white=False,
    )
    eval_black += queen_handcrafted_eval(
        black_piece_map["q"],
        white_piece_position_stringmap,
        black_diag_map,
        diag_pos_map,
    )
    return eval_white - eval_black


def pawn_handcrafted_eval(pawn_list, player_position_map, opp_position_map, is_white):
    diff = 0
    advancement_pawn = {
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 3,
        "5": 5,
        "6": 13,
        "7": 34,
        "8": 900,
    }
    for position in pawn_list:
        if player_position_map["p"].count(position[0]) > 1:
            # doubled pawns
            diff -= 7
        if (
            not str(int(position[1]) - 1) in player_position_map["p"]
            or not str(int(position[1]) + 1) in player_position_map["p"]
        ):
            # isolated pawn
            diff -= 2
        if position[0] not in opp_position_map["p"]:
            # passed pawn
            diff += 1
        if is_white:
            diff += advancement_pawn[position[1]]
        else:
            diff += advancement_pawn[str(9 - int(position[1]))]
    return diff


def bishop_handcrafted_eval(bishop_list, player_diag_map, diag_pos_map):
    diff = 0
    bishop_pair = 30
    if len(bishop_list) == 2:
        # Bishop Pair
        diff += bishop_pair
        diags = diag_pos_map[bishop_list[0]]
        diags += diag_pos_map[bishop_list[1]]
    elif len(bishop_list) == 0:
        return 0
    else:
        diags = diag_pos_map[bishop_list[0]]
    for diag in diags:
        # if same diagonal as a same color pawn, reduce diff
        if "p" in player_diag_map[diag]:
            diff -= 15
    return diff


def knight_handcrafted_eval(knight_list):
    diff = 0
    return diff


def rook_handcrafted_eval(rook_list, player_position_map, opp_position_map, is_white):
    diff = 0
    file_dic = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}
    bonuses_king_proximity = {0: 15, 1: 9, 2: 5, 3: 4, 4: 3, 5: 1, 6: -1, 7: -3}
    if len(rook_list) == 2 and (
        rook_list[0][0] == rook_list[1][0] or rook_list[0][1] == rook_list[1][1]
    ):
        # Connected rooks
        diff += 15
    for rook in rook_list:
        min_dist = min(
            abs(file_dic[rook[0]] - file_dic[opp_position_map["k"][0]]),
            abs(int(rook[1]) - int(opp_position_map["k"][1])),
        )
        diff += bonuses_king_proximity[min_dist]
        # 3 bonus if no friendly pawns in front and enemy 10 bonus if no pawns in front
        if (
            rook[0] not in player_position_map["p"]
            and rook[0] not in opp_position_map["p"]
        ):
            diff += 10
        elif (
            rook[0] not in player_position_map["p"] and rook[0] in opp_position_map["p"]
        ):
            diff += 3
        # 20 bonus if rank == 7
        if "7" in rook and is_white:
            diff += 20
        elif not is_white and "2" in rook:
            diff += 20
    return diff


def queen_handcrafted_eval(queen_list, opp_position_map, player_diag_map, diag_pos_map):
    diff = 0
    file_dic = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}
    bonuses_king_proximity = {0: 15, 1: 9, 2: 5, 3: 4, 4: 3, 5: 1, 6: -1, 7: -3}
    # if same diagonal as bishop + x points
    diags = []
    for queen in queen_list:
        diags += diag_pos_map[queen]
        diff += bonuses_king_proximity[
            abs(file_dic[queen[0]] - file_dic[opp_position_map["k"][0]])
        ]
        diff += bonuses_king_proximity[
            abs(int(queen[1]) - int(opp_position_map["k"][1]))
        ]
    for diag in diags:
        if "b" in player_diag_map[diag]:
            diff += 15
    # distance from enemy king is awarded

    return diff
