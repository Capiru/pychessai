import chess as ch
import matplotlib.pyplot as plt
import chess.svg as svg
import cairosvg
import matplotlib.image as mpimg

def show_board(board,eval_white,eval_black,fig = None):
    if fig is None:
        fig = plt.figure(figsize=(5,5))
        plt.ion()
        plt.show()
    svg_img = svg.board(board=board)
    f = open('tmp/board.svg', 'w')
    f.write(svg_img)
    f.close()
    cairosvg.svg2png(url='tmp/board.svg', write_to='tmp/image.png')
    img = mpimg.imread('tmp/image.png')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    txt = plt.text(0.05, 0.95, f"White: {eval_white:.3f}\nBlack: {eval_black:.3f}", fontsize=14,
        verticalalignment='top', bbox=props)
    plt.imshow(img)
    plt.pause(0.01)
    txt.set_visible(False)
    return fig

# board = ch.Board()
# fig = show_board(board,-0.001,0.3)
# board.push_san(san="e4")
# show_board(board,0,0,fig)