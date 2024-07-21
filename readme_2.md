# Files
    foundpose_3.ipynb - full pipeline for foundpose 
    foundpose_check_code.ipynb - check feaures extraction from dinov2d
    cnos_analysis_3.ipynb - code for rendering using Blenderproc in cnos
    cnos_analysis_4.ipynb - code for checking cnos code for mutiple objects - from 2 objects to see if the features work like top images are from correct object
    cnos_analysis_5.ipynb - final code check for cnos after 23th layer with all cases , occulsion, etc 
    cnos_analysis_5.ipynb - final code check for cnos after 18th layer with all cases , occulsion, etc 
    approach_first_check.ipynb: 
        Check the first approach by extracting features from dinov2_l14 at the last layer then retrieve the templates with highest score - see if the templates has similar poses to the input  
    approach_second_check.ipynb: 
        Extracting fatures from dinov2_l14_reg get the patches features instead then still match the patches descriptor -to get the templates with similar poses

    foundpose_BoW_check.ipynb - check BoW code
# Create templates from train_pbr (BlenderProc) or from test (real images)
    In cnos_analysis_4.ipynb - output folder will be in foundpose_analysis folder
    fro real images- use lower templates since we have less images - just 162 are enough I believe
# Create templates from CAD models 
    python -m src.scripts.render_template_with_pyrender level=2 # 0 is for 42 templates, 3 for 2562 templates
    Templates will be save in the folder templates_pyrender
    10* that [50:450, 150:500, :3] for templates will return better zoomed-in templates - Do that

# Run
Download all dataset
    python -m src.scripts.download_bop23
Render templates with pyrender for all dataset
    python -m src.scripts.render_template_with_pyrender level=2 # 0 is for 42 templates, 3 for 2562 templates
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

Testing on custom image
    export CAD_PATH=datasets/bop23_challenge/datasets/xyz/models/obj_000006.ply
    export RGB_PATH=datasets/bop23_challenge/datasets/xyz/banjinjian/000001/rgb/000004.png
    export OUTPUT_DIR=./tmp/custom_dataset

    # render templates from cad models
    bash src/scripts/render_custom.sh

    bash src/scripts/run_inference_custom.sh

# Testing on dataset xyz
    export CAD_PATH=datasets/bop23_challenge/datasets/xyz/models/obj_000006.ply
    export RGB_PATH=datasets/bop23_challenge/datasets/xyz/banjinjian/000001/rgb/000004.png
    export OUTPUT_DIR=./tmp/custom_dataset

    # render templates from cad models
    bash src/scripts/render_custom_xyz.sh

    bash src/scripts/run_inference_custom.sh





