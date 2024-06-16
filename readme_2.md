Download all dataset
    python -m src.scripts.download_bop23
Render templates with pyrender for all dataset
    python -m src.scripts.render_template_with_pyrender level=0 # 0 is for 42 templates
download model weights of SAM and Fast SAM
    python -m src.scripts.download_sam
    python -m src.scripts.download_fastsam
Download BlenderProc4BOP set - 10* This is only required when you want to use realistic rendering with BlenderProc.
    For BOP challenge 2023 core datasets (LMO, TLESS, TUDL, ICBIN, ITODD, HB, and TLESS):
        python -m src.scripts.download_train_pbr

Testing on BOP dataset
    export DATASET_NAME=icbin 
    FAST_SAM and pbr
        python run_inference.py dataset_name=$DATASET_NAME model=cnos_fast

    with SAM + pyrender
        python run_inference.py dataset_name=$DATASET_NAME model.onboarding_config.rendering_type=pyrender

    with SAM + PBR
        python run_inference.py dataset_name=icbin

Visulizing the results
    export DATASET_NAME=icbin 
    export INPUT_FILE=datasets/bop23_challenge/results/cnos_exps/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin.json

    export OUTPUT_DIR=datasets/bop23_challenge/results/cnos_exps/visualization/sam_pbr_icbin_20%/
    # normal visulization
    python -m src.scripts.visualize dataset_name=$DATASET_NAME input_file=$INPUT_FILE output_dir=$OUTPUT_DIR

    # visulize with score, value
    python -m src.scripts.visualize_detectron2 dataset_name=$DATASET_NAME input_file=$INPUT_FILE output_dir=$OUTPUT_DIR





