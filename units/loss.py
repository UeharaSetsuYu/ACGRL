import torch
import torch.nn as nn
import torch.nn.functional as F




class ClusterContrastiveLoss(nn.Module):
    def __init__(self, class_num, temperature=0.5):
        super().__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def similarity(self, a, b): 
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.matmul(a.squeeze(1), b.squeeze(0).T)

    def mask_correlated_samples(self, N): 
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(self.class_num):
            mask[i, self.class_num + i] = False
            mask[self.class_num + i, i] = False
        return mask

    def forward(self, q_i, q_j): 
        eps = 1e-10    
        p_i = q_i.sum(0)  # [C]
        p_i = p_i / (p_i.sum() + eps)
        p_j = q_j.sum(0)  # [C]
        p_j = p_j / (p_j.sum() + eps)
 
        ne_i = torch.log(torch.tensor(self.class_num, dtype=torch.float32, device=q_i.device)) \
               + (p_i * torch.log(p_i + eps)).sum()
        ne_j = torch.log(torch.tensor(self.class_num, dtype=torch.float32, device=q_i.device)) \
               + (p_j * torch.log(p_j + eps)).sum()
        entropy = ne_i + ne_j
 
        q_i = q_i.t()  # [C, B]
        q_j = q_j.t()  # [C, B]
        N = 2 * self.class_num
        q = torch.cat([q_i, q_j], dim=0)  # [2C, B]
 
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature  # [2C, 2C]
 
        sim_i_j = torch.diag(sim, self.class_num)  # [C]
        sim_j_i = torch.diag(sim, -self.class_num)  # [C]
        positive_clusters = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)  # [2C,1]
 
        mask = self.mask_correlated_samples(N).to(q.device)
        negative_clusters = sim[mask].reshape(N, -1)  # [2C, 2C-2]
 
        logits = torch.cat([positive_clusters, negative_clusters], dim=1)  # [2C, 1+(2C-2)]
        labels = torch.zeros(N, dtype=torch.long, device=q.device)  # 全部是正样本索引 0
        loss = self.criterion(logits, labels) / N
 
        return loss + entropy



class Loss(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.pseudoAlignLoss = ClusterContrastiveLoss(class_num)
        self.mse = nn.MSELoss()
        self.crossentropy = nn.CrossEntropyLoss()

    def forward_mse(self, x, y):
        return self.mse(x, y)

    def forward_CrossEntropy(self, logits, labels):
        return self.crossentropy(logits, labels)

    @staticmethod
    def forward_Entropy(logits, eps=1e-8):
        probabilities = F.softmax(logits, dim=1)
        return -(probabilities * torch.log(probabilities + eps)).sum(dim=1).mean()


