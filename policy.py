import torch
import chess as ch
from config import CFG

def map_policy_to_move(action,board):
    ### takes an int and converts it into a legal move
    vector = torch.zeros((73,8,8))
    return_move_list = []
    original_action = get_original_coordinates(action)
    legal_moves = board.legal_moves
    for move in legal_moves:
        initial_square = move.from_square
        piece_type = board.piece_at(initial_square)
        to_square = move.to_square
        if str(piece_type).lower() == "n" or piece_type == ch.KNIGHT:
            plane_no = 56
            plane_no += get_plane_knight(initial_square,to_square)
        elif move.promotion is not None and not move.promotion == ch.QUEEN:
            plane_no = 64
            plane_no += get_plane_underpromotion(initial_square,to_square,move)
        else:
            plane_no = 0
            plane_no += get_plane_queen_moves(initial_square,to_square)
        vector[plane_no,initial_square%8,((initial_square)//8)%8] = 1
        if plane_no == original_action[0] and initial_square%8 == original_action[1] and ((initial_square)//8)%8 == original_action[2]:
            return move
    return None

def map_moves_to_policy(legal_moves,board,flatten = False,dic = None):
    ### Returns a vector same sized as policy with all legal moves, with 1 on possible legal moves
    vector = torch.zeros((73,8,8))
    return_move_list = []
    for move in legal_moves:
        initial_square = move.from_square
        piece_type = board.piece_at(initial_square)
        to_square = move.to_square
        if str(piece_type).lower() == "n" or piece_type == ch.KNIGHT:
            plane_no = 56
            plane_no += get_plane_knight(initial_square,to_square)
        elif move.promotion is not None and not move.promotion == ch.QUEEN:
            plane_no = 64
            plane_no += get_plane_underpromotion(initial_square,to_square,move)
        else:
            plane_no = 0
            plane_no += get_plane_queen_moves(initial_square,to_square)
        if dic is not None:
            vector[plane_no,initial_square%8,((initial_square)//8)%8] = dic[move]
        else:
            vector[plane_no,initial_square%8,((initial_square)//8)%8] = 1
        return_move_list.append(get_vector_coord(plane_no,initial_square%8,((initial_square)//8)%8))
        if CFG.DEBUG:
            flat = torch.flatten(vector)
            index = (flat == 1).nonzero(as_tuple=True)[0]
            a,b,c = get_original_coordinates(index)
            print(index,a,b,c,plane_no,initial_square%8,((initial_square)//8)%8)
            assert vector[a,b,c] == 1
            assert get_vector_coord(a,b,c) == index
            print(get_vector_coord(a,b,c))
            vector[plane_no,initial_square%8,((initial_square)//8)%8] = 0

    if flatten:
        return torch.flatten(vector),return_move_list
    else:
        return vector

def is_move_legal(move,board):
    actions,_ = map_moves_to_policy(board.legal_moves,board,flatten = True)
    if actions[move] == 1:
        return True
    else:
        return False

def get_original_coordinates(pos):
    ### Takes a position from a 4672 vector and turns it back into (x,y,z) in a (73x8x8) tensor
    a = int(pos//(8*8))
    b = pos-64*a
    return (a,int(b//(8)), int((b)%8))

def get_vector_coord(a,b,c):
    ### Takes a tensor coord (a,b,c) from a (73x8x8) tensor and turns into a coordinate in a 4672 tensor
    return a*64 + b*8 + c

def get_horiz_vert(pos):
    ### Turns a 0 to 63 numbe into board coordinates [0-7,0-7]
    return pos%8,((pos)//8)%8

def get_distances(init_vec,to_vec):
    return to_vec[0]-init_vec[0],to_vec[1]-init_vec[1]

def get_plane_knight(initial_square,to_square):
    init_hor,init_ver = get_horiz_vert(initial_square)
    to_hor,to_ver = get_horiz_vert(to_square)
    horizontal_dist,vertical_dist = get_distances([init_hor,init_ver],[to_hor,to_ver])
    possible_moves = {"1,2":0,"2,1":1,"-1,2":2,"2,-1":3,"-1,-2":4,"-2,-1":5,"1,-2":6,"-2,1":7}
    return possible_moves[f"{horizontal_dist},{vertical_dist}"]

def get_plane_underpromotion(initial_square,to_square,move):
    init_hor,init_ver = get_horiz_vert(initial_square)
    to_hor,to_ver = get_horiz_vert(to_square)
    horizontal_dist,vertical_dist = get_distances([init_hor,init_ver],[to_hor,to_ver])
    if move.promotion == ch.KNIGHT:
        skip = 0
        return skip*3 + 1 + horizontal_dist
    elif move.promotion == ch.BISHOP:
        skip = 1
        return skip*3 + 1 + horizontal_dist
    else:
        skip = 2
        return skip*3 + 1 + horizontal_dist

def get_plane_queen_moves(initial_square,to_square):
    init_hor,init_ver = get_horiz_vert(initial_square)
    to_hor,to_ver = get_horiz_vert(to_square)
    horizontal_dist,vertical_dist = get_distances([init_hor,init_ver],[to_hor,to_ver])
    possible_moves = {"1,0":0,"2,0":1,"3,0":2,"4,0":3,"5,0":4,"6,0":5,"7,0":6,
                      "-1,0":7,"-2,0":8,"-3,0":9,"-4,0":10,"-5,0":11,"-6,0":12,"-7,0":13,
                      "0,1":14,"0,2":15,"0,3":16,"0,4":17,"0,5":18,"0,6":19,"0,7":20,
                      "0,-1":21,"0,-2":22,"0,-3":23,"0,-4":24,"0,-5":25,"0,-6":26,"0,-7":27,
                      "1,1":28,"2,2":29,"3,3":30,"4,4":31,"5,5":32,"6,6":33,"7,7":34,
                      "-1,1":35,"-2,2":36,"-3,3":37,"-4,4":38,"-5,5":39,"-6,6":40,"-7,7":41,
                      "1,-1":42,"2,-2":43,"3,-3":44,"4,-4":45,"5,-5":46,"6,-6":47,"7,-7":48,
                      "-1,-1":49,"-2,-2":50,"-3,-3":51,"-4,-4":52,"-5,-5":53,"-6,-6":54,"-7,-7":55,}
    return possible_moves[f"{horizontal_dist},{vertical_dist}"]