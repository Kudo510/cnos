import torch

from src.model.constrastive_learning import ContrastiveModel, train
from src.model.constrastive_learning import extract_dataset

dataset="icbin"
data_type="test"
scene_id=1
pos_proposals, neg_proposals = extract_dataset(dataset, data_type, scene_id) # Take 2.21 minutes

all_pos_proposals = [item for sublist in pos_proposals for item in sublist]
all_neg_proposals = [item for sublist in neg_proposals for item in sublist]

template_paths = "datasets/bop23_challenge/datasets/templates_pyrender/icbin_720/obj_obj_000001/*.png"

template_poses_path = "datasets/bop23_challenge/datasets/templates_pyrender/icbin_720/obj_poses.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveModel(device)

train(device = device, model= model, template_paths=template_paths, template_poses_path=template_poses_path,
    all_pos_proposals=all_pos_proposals, all_neg_proposals=all_neg_proposals)