import torch

from src.model.constrastive_learning import ContrastiveModel, train
from src.model.constrastive_learning import extract_dataset

dataset="icbin"
data_type="test"
scene_id=1
pos_proposals, neg_proposals = extract_dataset(dataset, data_type, scene_id) # Take 2.21 minutes

all_pos_proposals = [item for sublist in pos_proposals for item in sublist]
all_neg_proposals = [item for sublist in neg_proposals for item in sublist]

import pickle

with open('contrastive_learning/all_pos_proposals.pkl', 'wb') as file:
    pickle.dump(all_pos_proposals, file)

with open('contrastive_learning/all_neg_proposals.pkl', 'wb') as file:
    pickle.dump(all_neg_proposals, file)