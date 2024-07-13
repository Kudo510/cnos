from torch import nn
import torch
from src.poses.utils import load_rotation_transform, convert_openCV_to_openGL_torch
import torch.nn.functional as F
from src.model.utils import BatchedData


class Similarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(Similarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference): # query is stack of proposals # reference is the features from templates
        query = F.normalize(query, dim=-1) 
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


class PairwiseSimilarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        '''
        E.g for ycbv we have 21 objects
        query shape: (56, 1024) = num_proposal/N_query, features_dim
        reference shape: (21, 42, 1024) = num_obj, num_templates, features_dim
        goal to convert to 
            query shape : num_proposal/N_query, num_templates, features_dim
            reference shape: size of N_query, num_obj, num_templates, features_dim
        Then for each obj_id - normalize and compare the cosine simlarity of 2 vector N_query, num_templates, features_dim- return N_query, num_templates
        Then stack all the similarity of all ob_ids- we get num_obj, N_query, num_templates
        Permute to get the size of N_query x N_objects x N_templates
        '''
        N_query = query.shape[0]
        N_objects, N_templates = reference.shape[0], reference.shape[1]
        references = reference.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)  # N_query, num_obj, num_templates, features_dim
        queries = query.clone().unsqueeze(1).repeat(1, N_templates, 1) # num_proposal/N_query, num_templates, features_dim
        queries = F.normalize(queries, dim=-1)
        references = F.normalize(references, dim=-1)

        similarity = BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim = F.cosine_similarity(
                queries, references[:, idx_obj], dim=-1
            )  # N_query x N_templates
            similarity.append(sim)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2)  # N_query x N_objects x N_templates
        return similarity.clamp(min=0.0, max=1.0)
