import torch
from torch import nn
class Regressor(nn.Module):
    """This manages the output. I follow a similar structure to this paper but make a few modifications (for the better or worse) 
    https://arxiv.org/abs/1712.01815
    Basically this allows for the network not only to learn the best move, but the legality of moves as well" 
    """
    """the first part of the output is 16 neurons that represent the piece to move, the second part of the output will be reshaped to be (16,8,8). 
    Where the first 16 represent pieces on the board, and the 8x8 is the board itself. 
    The only issue is duplicate pieces, which I try to address by finding the first legal move out of the pieces. 
    #todo 
    This doesn't deal with promotions,resign, draw at all yet."""

    def __init__(self, inputs):
        super(Regressor, self).__init__()
        self.layer = nn.Linear(inputs, (16 * 8 * 8) + 16 )
    def forward(self, x):
        return self.layer(x)

class UniversalChessInterface(nn.Module):
    """Take the input from the network, process it, and make it in the Universal Chess Interface
        https://en.wikipedia.org/wiki/Universal_Chess_Interface 
        https://wbec-ridderkerk.nl/html/UCIProtocol.html
    """
    def to_int(self,input):
        out = torch.argmax(self.softmax(input))
        return out, self.piece[out]
    """Check if move written in LAN is legal""" 
    def is_legal(self,LAN):

        
        return False
    def __init__(self, piece_dict={
        0: "",  # pawn
        1: "",  # pawn
        2: "",  # pawn
        3: "",  # pawn
        4: "",  # pawn
        5: "",  # pawn
        6: "",  # pawn
        7: "",  # pawn
        8: "",  # pawn
        9: "K", 
        10: "Q",
        11: "R",
        12: "R",
        13: "B",
        14: "B",
        15: "N",
        16: "N",
    },position_dict={
        0:"a",1:"b",2:"c",3:"d",4:"e",5:"f", 6:"g", 7:"h"
    }):
        super(UniversalChessInterface, self).__init__()
        self.softmax = nn.Softmax()
        self.piece = piece_dict
        self.converter = self.to_int 
        self.position_dict = position_dict
    def forward(self, x,original_board):
        possible_moves, piece= self.converter(x[0:16])
        move_prob = x[:16]
        x = x[16:].view(16,8,8)
        board =  original_board
        legal_moves = list(board.legal_moves)
        for i in range(16):
            pos = (x[possible_moves]==torch.max(x[possible_moves])).nonzero()
            file,rank = str(self.position_dict(pos[0].item())),str(pos[1].item())
            san = str(piece +file + rank)
            if(san in legal_moves):
                return san
            move_prob[possible_moves] = -1
            possible_moves, piece= self.converter(move_prob)
        return "resign"