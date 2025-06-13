import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from data_preprocess import FeaturesExtractor
from torch_geometric.nn import GATConv, global_mean_pool
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'



class GraphNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim, hidden_dim)
        #GAT_Layers
        self.gat_layers = nn.ModuleList([GATConv(hidden_dim,  hidden_dim // num_heads,heads=num_heads,dropout=dropout) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.global_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embedding(x)
        h = self.dropout(h)
        if hasattr(data, 'node_types'):
            node_type_embed = self.node_type_embedding(data.node_types)
            h = h + node_type_embed
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)
            h = h + h_prev  #residual connection
            h = F.relu(h)
            h = self.dropout(h)
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        h_graph = torch.sum(h * attention_weights, dim=0)
        h_graph = h_graph + global_mean_pool(h, batch)
        return h_graph

class SetTransformer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4),nn.ReLU(),nn.Dropout(dropout),nn.Linear(dim * 4, dim))
        
    def forward(self, x1, x2):
        attended, _ = self.attention(x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0))
        attended = attended.squeeze(0)
        x = x1 + attended
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class MomentumLearning(nn.Module):
    def __init__(self, hidden_dim, temperature=0.07, momentum=0.999, queue_size=4096):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size
        self.encoder_q = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim))
        self.encoder_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.register_buffer("queue", torch.randn(hidden_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:     #Replace oldest keys
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:  #Handle queue wraparound
            remainder = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remainder].T
            self.queue[:, :batch_size-remainder] = keys[remainder:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
        
    def forward(self, x1, x2, labels=None):
        q = F.normalize(self.encoder_q(x1), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(x2), dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)    #InfoNCE loss
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels_contrastive = torch.zeros(logits.shape[0], dtype=torch.long, device=x1.device)
        if labels is not None:  # Hard negative mining (optional)
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
            mask.fill_diagonal_(False)
            mask = mask.float()
            logits_mask = torch.ones_like(mask)
            logits_mask.fill_diagonal_(0)
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -mean_log_prob_pos.mean()
        else:
            #Original InfoNCE loss
            loss = F.cross_entropy(logits, labels_contrastive)
        self._dequeue_and_enqueue(k)
        return loss

class DataAugmentation:
    def __init__(self, edge_drop_rate=0.1, node_mask_rate=0.1, feature_mask_rate=0.1):
        self.edge_drop_rate = edge_drop_rate
        self.node_mask_rate = node_mask_rate
        self.feature_mask_rate = feature_mask_rate
        
    def augment_graph(self, data):
        aug_data = Data()
        for key, item in data:
            aug_data[key] = item
        if self.edge_drop_rate > 0:
            num_edges = aug_data.edge_index.size(1)
            num_drop = int(num_edges * self.edge_drop_rate)
            drop_idx = torch.randperm(num_edges)[:num_drop]
            mask = torch.ones(num_edges, dtype=torch.bool)
            mask[drop_idx] = False
            aug_data.edge_index = aug_data.edge_index[:, mask]
            if hasattr(aug_data, 'edge_attr') and aug_data.edge_attr is not None:
                aug_data.edge_attr = aug_data.edge_attr[mask]
        
        if self.feature_mask_rate > 0 and hasattr(aug_data, 'x'):
            num_nodes, num_features = aug_data.x.size()
            mask = torch.rand((num_nodes, num_features)) > self.feature_mask_rate
            aug_data.x = aug_data.x * mask.to(aug_data.x.device)
        return aug_data
    

           
class MolecularNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, 
                 num_node_types, num_edge_types, processing_steps, use_Momentum=True):
        super().__init__()
        self.graph_processor = GraphNetwork( in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, 
                                                            num_edge_types, processing_steps)
        self.features_processor =  FeaturesExtractor(hidden_dim, dropout)
        self.cross_attention = SetTransformer(hidden_dim, num_heads, dropout)
        self.augmentation = DataAugmentation(edge_drop_rate=0.1,node_mask_rate=0.1,feature_mask_rate=0.1 )
        self.use_Momentum = use_Momentum
        if use_Momentum:
            self.contrastive_learner = MomentumLearning(hidden_dim=hidden_dim,temperature=0.07,momentum=0.999,queue_size=4096)
        self.graph_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim, 1))
        self.combined_classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim, 1))
        
    def forward(self, data, features_batch=None, return_embeddings=False):
        graph_features = self.graph_processor(data)
        if features_batch is not None:
            add_features = self.features_processor(features_batch)
            combined_features = self.cross_attention(graph_features, add_features)
            out = self.combined_classifier(torch.cat([graph_features, combined_features], dim=-1))
        else:
            out = self.graph_classifier(graph_features)
        if return_embeddings:
            return out.squeeze(-1), graph_features
        else:
            return out.squeeze(-1)
    
    def contrastive_forward(self, data, features_batch=None, labels=None):
        if not self.use_Momentum:
            raise RuntimeError("Contrastive learning is disabled for this model")
        aug_data1 = self.augmentation.augment_graph(data)
        aug_data2 = self.augmentation.augment_graph(data)
        _, graph_features1 = self.forward(aug_data1, return_embeddings=True)
        _, graph_features2 = self.forward(aug_data2, return_embeddings=True)
        contrastive_loss = self.contrastive_learner(graph_features1, graph_features2, labels)
        return contrastive_loss
    
    def training_step(self, data, features_batch=None, labels=None, alpha=0.5):
        logits = self.forward(data, features_batch)
        if labels is not None:
            supervised_loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            supervised_loss = 0
        if self.use_Momentum:
            contrastive_loss = self.contrastive_forward(data, features_batch, labels)
            # Combine losses with weighting factor alpha
            total_loss = supervised_loss + alpha * contrastive_loss
            return total_loss, supervised_loss, contrastive_loss
        else:
            return supervised_loss

