# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley
# Date:    10/11/2019
# Purpose: Model for solving RPM problems

# IMPORTS ----------------------------------------------------------------------------------------------------------- #

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   basic_model import BasicModel

# CONSTANTS --------------------------------------------------------------------------------------------------------- #

DR_S, DR_F =  .1, .5      #Dropout prob. for spatial and fully-connected layers.
O_HC, O_OC =  64, 64      #Hidden and output channels for original enc.
F_HC, F_OC =  64, 16      #Hidden and output channels for frame enc.
S_HC, S_OC = 128, 64      #Hidden and output channels for sequence enc.
F_PL, S_PL = 5*5, 16      #Pooled sizes for frame and sequence enc. outputs.
F_Z = F_OC*F_PL           #Frame embedding dimensions.
K_D = 7                   #Conv. kernel dimensions.

BL_IN = 3
BLOUT = F_Z
G_IN  = BLOUT
G_HID = G_IN
G_OUT = G_IN
R_OUT = 32
C_DIM = 2
P_DIM = 32
C     = 1.0
    
# CLASSES ----------------------------------------------------------------------------------------------------------- #

#Helper function for spatial dropout.
class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)
    
class flat(nn.Module):
    def __init__(self):
        super(flat, self).__init__()
    def forward(self, x):
        return x.flatten(1)
    
#Convolutional block class (conv, elu, bnorm, dropout). If 1D block, no downsampling. If 2D, stride==2.
#Implements spatial dropout for both 1D and 2D convolutional layers.
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv  = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, K_D, stride=dim, padding=K_D//2)
        self.bnrm  = getattr(nn, 'BatchNorm{}d'.format(dim))(out_ch)
        self.drop  = nn.Sequential(perm(), nn.Dropout2d(DR_S), perm()) if dim==1 else nn.Dropout2d(DR_S)
        self.block = nn.Sequential(self.conv, nn.ELU(), self.bnrm, self.drop)
    def forward(self, x):
        return self.block(x)

#Residual block class, made up of two convolutional blocks.
class ResBlock(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock, self).__init__()
        self.dim  = dim
        self.conv = nn.Sequential(ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim))
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim==1 else self.down(x))

#Relational net class, defining three models with which to extract relations in RPM problems.
class RelNet(nn.Module):
    def __init__(self, model, n_s):
        super(RelNet, self).__init__()
        self.stack  = lambda x : torch.stack([torch.cat((x[:,:8], x[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], dim=1)
        self.model  = model
        self.n_s    = n_s
               
        if model in ["Rel-AIR", "Rel-Base"]:
            lin_in = S_OC*S_PL
            self.obj_enc = nn.Sequential(ResBlock(   1, F_HC, F_HC, 2), ResBlock(F_HC, F_HC, F_OC, 2))
            self.seq_enc = nn.Sequential(ResBlock(   9, S_OC, S_HC, 1), nn.MaxPool1d(6, 4, 1), 
                                         ResBlock(S_HC, S_HC, S_OC, 1), nn.AdaptiveAvgPool1d(S_PL))
            if model in ["Rel-AIR"]:
                self.obj_rel   = nn.Sequential(ResBlock(n_s, S_HC, S_HC, 1), ResBlock(S_HC, S_HC, 1, 1))
                self.bilinear  = nn.Bilinear(F_Z, BL_IN, BLOUT)
                self.ebd       = nn.Sequential(nn.ELU(), nn.BatchNorm1d(BLOUT))

        elif model in ["ResNet", "Context-blind"]:
            lin_in = O_OC*F_PL
            self.og_net = nn.Sequential(ResBlock(9 if model=="ResNet" else 8, O_HC, O_HC, 2), ResBlock(O_HC, O_HC, O_OC, 2))
            
        else:
            print("Model \"{}\" unrecognised.".format(model))
            sys.exit(1)
                    
        self.linear  = nn.Sequential(nn.Linear(lin_in, 512), nn.ELU(), nn.BatchNorm1d(512), nn.Dropout(DR_F), 
                                     nn.Linear(512, 8 if model=='Context-blind' else 1))

    def forward(self, x, n_s):                
        if self.model in ['Rel-GNN', 'Rel-AIR']:
            obj,pos,n = x

            #1. Encode each object, of each frame, of each RPM, separately.
            x = self.obj_enc(obj.view(-1, 1, 80, 80)).flatten(1)
            x = x.view(-1, 16, n_s, F_Z)
            
            #2. Feed object embeddings and pos data through a bilinear layer.
            x = x.view(-1, F_Z)
            p = pos.view(-1, 3)
            x = self.ebd(self.bilinear(x, p))
                
            #3. Perform feature extraction for object relationships using residual blocks.                    
            x = x.view(-1, 16, n_s, G_IN)
            x = self.obj_rel(x.view(-1, n_s, G_IN)).view(-1, 16, G_IN)
            
            #4. Assemble RPM into 8 sequences of 9 frames each (8 context + 1 answer). 
            #   Alternatively, assemble into 10 sequences of 3 frames each (2 problem rows + 8 candidate rows).
            x = self.stack(x)
           
            #5. Perform feature extraction for frame relationships, and score sequences.
            x = self.seq_enc(x.view(-1, 9, G_OUT if self.model=='Rel-GNN' else G_IN)).flatten(1)
            return self.linear(x).view(-1, 8)
        
        elif self.model=='Rel-Base':
            #1. Encode each frame independently.
            x = x.view(-1, 1, 80, 80)
            x = self.obj_enc(x).flatten(1)
            
            #2. Assemble sequences. 
            x = x.view(-1, 16, F_Z)
            x = self.stack(x)
                        
            #3. Extract frame relationships and score sequences.
            x = self.seq_enc(x.view(-1, 9, F_Z)).flatten(1)
            return self.linear(x).view(-1, 8)
        
        elif self.model=='ResNet':
            #1. Assemble sequences, extract frame relationships, score sequences.
            x = self.stack(x)
            x = self.og_net(x.view(-1, 9, 80, 80)).flatten(1)
            return self.linear(x).view(-1, 8)
        
        else:
            x = self.og_net(x[:, 8:])
            return self.linear(x.flatten(1))
            
      
#Main model class.
class RPM_Solver(BasicModel):
    def __init__(self, args):
        super(RPM_Solver, self).__init__(args)
        self.model     = args.model
        self.rel_enc   = nn.DataParallel(RelNet(args.model, args.trn_n), device_ids=[0,1]) \
                         if args.multi_gpu else RelNet(args.model, args.trn_n)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        
    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)
    
    def forward(self, x, n_s):
        x = x if self.model=='Rel-AIR' else 1 - x/255.0
        out = self.rel_enc(x, n_s)
        return out

# END SCRIPT -------------------------------------------------------------------------------------------------------- #
