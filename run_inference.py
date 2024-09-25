import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    os.makedirs(cfg.callback.checkpoint.dirpath, exist_ok=True)
    logging.info(
        f"Training script. The outputs of hydra will be stored in: {output_path}"
    )
    logging.info("Initializing logger, callbacks and trainer")

    if cfg.machine.name == "slurm": # not our case
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        logging.info(f"Slurm config: {num_gpus} gpus,  {num_nodes} nodes")
    trainer = instantiate(cfg.machine.trainer) # is pytorch_lightning.Trainer with config as configs/machine/trainer/local.yaml

    default_ref_dataloader_config = cfg.data.reference_dataloader
    default_query_dataloader_config = cfg.data.query_dataloader

    query_dataloader_config = default_query_dataloader_config.copy()
    ref_dataloader_config = default_ref_dataloader_config.copy()

    if cfg.dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"
    query_dataloader_config.root_dir += f"{cfg.dataset_name}" # ./datasets/bop23_challenge/datasets/tless
    # import pdb; pdb.set_trace()
    query_dataset = instantiate(query_dataloader_config) # src.dataloader.bop.BaseBOPTest for tless

    logging.info("Initializing model")
    model = instantiate(cfg.model) # src.model.detector.CNOS

    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names # list of [001_obj, oo2_obj,..., 028_obj] 
    model.dataset_name = cfg.dataset_name # tless

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,  # only support a single image for now
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )
    if cfg.model.onboarding_config.rendering_type == "pyrender": # our case pbr not pyrender

        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr": # here our case
        logging.info("Using BlenderProc for reference images")
        ref_dataloader_config._target_ = "src.dataloader.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = f"{query_dataloader_config.root_dir}"
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}" # ./datasets/bop23_challenge/datasets/templates_pyrender/icbin'
        ref_dataset = instantiate(ref_dataloader_config) # src.dataloader.bop.BOPTemplatePBR
        ref_dataset.load_processed_metaData(reset_metaData=True)
    else:
        raise NotImplementedError
    model.ref_dataset = ref_dataset # src.dataloader.bop.BOPTemplatePBR

    segmentation_name = cfg.model.segmentor_model._target_.split(".")[-1] # CNOS
    agg_function = cfg.model.matching_config.aggregation_function #avg_5 means average of top 5
    rendering_type = cfg.model.onboarding_config.rendering_type # pbr
    level_template = cfg.model.onboarding_config.level_templates # 0: which is coarse
    model.name_prediction_file = f"{segmentation_name}_template_{rendering_type}{level_template}_agg{agg_function}_{cfg.dataset_name}" # just the file name
    logging.info(f"Loading dataloader for {cfg.dataset_name} done!")
    trainer.test(
        model,
        dataloaders=query_dataloader,
    )
    logging.info(f"---" * 20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
