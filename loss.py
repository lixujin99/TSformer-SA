import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = torch.tensor(temperature)
        
    def forward(self, emb_i, emb_j):	
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        batch_size = emb_i.shape[0]
        negatives_mask = 1 - torch.eye(batch_size * 2, batch_size * 2, dtype=bool).float()

        representations = torch.cat([z_i, z_j], dim=0)          #(2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = (negatives_mask).to(emb_i.device) * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))     # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    
    

