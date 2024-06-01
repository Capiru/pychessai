from pychessai.utils.board_utils import get_players_piece_maps


def get_sorted_move_list(board, agent=None, only_attacks=False):
    # Execution time: 0.001401
    if agent is None:
        checkmate_list = []
        check_list = []
        capture_list = []
        attack_list = []
        # TODO: pin_list = []
        castling_list = []
        other_list = []
        move_list = list(board.legal_moves)
        w_map, b_map = get_players_piece_maps(board)
        for move in move_list:
            board.push(move)
            if board.is_checkmate():
                checkmate_list.append(move)
                board.pop()
                return checkmate_list
            elif board.is_check():
                check_list.append(move)
            board.pop()
            if board.is_capture(move):
                capture_list.append(move)
            elif only_attacks:
                other_list.append(move)
            elif board.is_castling(move):
                castling_list.append(move)
            else:
                other_list.append(move)
                attacks = board.attacks(move.to_square)
                if attacks:
                    if not bool(board.turn):
                        # white to play
                        if attacks.intersection(b_map):
                            attack_list.append(move)
                            other_list.pop()
                    else:
                        # black to play
                        if attacks.intersection(w_map):
                            attack_list.append(move)
                            other_list.pop()
        return_list = [
            *checkmate_list,
            *check_list,
            *capture_list,
            *attack_list,
            *castling_list,
            *other_list,
        ]
        if only_attacks:
            return [*checkmate_list, *check_list, *capture_list]
        else:
            return return_list
