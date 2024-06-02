import cairosvg
import chess as ch
import chess.svg as svg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
