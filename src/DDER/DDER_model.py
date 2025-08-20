import torch.nn as nn
import torch
import torch.nn.functional as F


class DDER(nn.Module):
    def __init__(self, num_views, dims, num_classes, device):
        super(DDER, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        self.device = device
        self.EvidenceCollectors = self.EvidenceCollectors.to(self.device)


    def Evidence_DC(self, alpha, beta):
        E = dict()
        for v in range(len(alpha)):
            E[v] = alpha[v]-1
            E[v] = torch.nan_to_num(E[v], 0)

        for v in range(len(alpha)):
            E[v] = torch.nan_to_num(E[v], 0)

        E_con = E[0]
        for v in range(1, len(alpha)):
            E_con = torch.min(E_con, E[v])
        for v in range(len(alpha)):
            E[v] = torch.sub(E[v], E_con)
        alpha_con = E_con + 1

        E_div = E[0]
        for v in range(1,len(alpha)):
            E_div = torch.add(E_div, E[v])
        E_div = torch.div(E_div, len(alpha))

        S_con = torch.sum(alpha_con, dim=1, keepdim=True)

        Sum0_con = torch.sum(E_con, dim=1, keepdim=True)
        E_S = torch.div(E_con, S_con)
        E_S = torch.pow(E_S, beta)
        E_con = torch.mul(E_S, S_con)
        Sum1_con = torch.sum(E_con, dim=1, keepdim=True)
        E_con = torch.mul(E_con, torch.div(Sum0_con,Sum1_con))


        E_con = torch.mul(E_con, len(alpha))
        E_a = torch.add(E_con, E_div)


        alpha_a = E_a + 1
        alpha_con = E_con + 1
        alpha_div = torch.add(E_div, 1)

        Sum = torch.sum(alpha_a, dim=1, keepdim=True)
        
        return alpha_a, alpha_con, alpha_div

    def forward(self, X, beta):
        # get evidence
        evidences = dict()
        pool_list = ["trans", "lstm", "attn"]
        for v in range(self.num_views):
           
            evidences[v] = self.EvidenceCollectors[v](X[v])
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidences[v_num] + 1
        alpha_a, alpha_con, alpha_div = self.Evidence_DC(alpha, beta)
        evidence_a = alpha_a - 1
        evidence_con = alpha_con - 1
        evidence_div = alpha_div - 1

        return evidences, evidence_a, evidence_con, evidence_div


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)

        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))

        self.net.append(nn.Softplus())
        
        
    def forward(self, x):

        if x.dim() == 3:

            x = torch.mean(x, dim=1)  

        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h









class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.att = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        scores = self.att(x)  # [batch_size, seq_len, 1]
        weights = F.softmax(scores, dim=1)  
        pooled = (x * weights).sum(dim=1)  
        return pooled
    



class MiniTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, max_heads=8, ff_dim=128, num_layers=1):
        super().__init__()

        for h in reversed(range(1, max_heads + 1)):
            if embed_dim % h == 0:
                num_heads = h
                break
        else:
            raise ValueError(f"No valid num_heads for embed_dim={embed_dim}")

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x): 
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0, :]  