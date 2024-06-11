import os

for dataset_name in [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",
        "hb",
        "ycbv",
    ]:
    # Fast SAM and PBR
    os.system(f"python run_inference.py dataset_name={dataset_name} model=cnos_fast")

    # With SAM + pyrender
    os.system(f"python run_inference.py dataset_name={dataset_name} model.onboarding_config.rendering_type=pyrender")

    # With SAM + PBR
    os.system(f"python run_inference.py dataset_name={dataset_name}")

    # Visualizing the results
    INPUT_FILE = f"datasets/bop23_challenge/results/cnos_exps/SAM_PBR_CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_{dataset_name}.json"
    OUTPUT_DIR = f"datasets/bop23_challenge/results/cnos_exps/visualization/sam_pbr_{dataset_name}/"
    os.system(f"python -m src.scripts.visualize dataset_name={dataset_name} input_file={INPUT_FILE} output_dir={OUTPUT_DIR}")
