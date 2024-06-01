def get_players_piece_maps(board):
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
