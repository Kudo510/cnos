# Files
    foundpose_3.ipynb - full pipeline for foundpose 
    foundpose_check_code.ipynb - check feaures extraction from dinov2d
    cnos_analysis_3.ipynb - code for rendering using Blenderproc in cnos
    cnos_analysis_4.ipynb - code for checking cnos code for mutiple objects - from 2 objects to see if the features work like top images are from correct object
    cnos_analysis_5.ipynb - final code check for cnos after 23th layer with all cases , occulsion, etc 
    cnos_analysis_6.ipynb - real final code ( using the test_an_image_step in cnos) - so use the inferece.py code not the custom one to test on single iamge- see the results look better now without any overlapping
    cnos_analysis_8.ipynb - cnos with contrastive learning - not compare features with cosin similarity - using NN for that instead- compare with 42 templates if the loss smaller than threshold means it is positive- out of 42 templates if there are a positive pair the proposal will be chosen

    constrastive_learning_3.ipynb

    cnos_foundpose for all 

    approach_first_check.ipynb: 
        Check the first approach by extracting features from dinov2_l14 at the last layer then retrieve the templates with highest score - see if the templates has similar poses to the input  
    approach_second_check.ipynb: 
        Extracting fatures from dinov2_l14_reg get the patches features instead then still match the patches descriptor -to get the templates with similar poses

    foundpose_BoW_check.ipynb - check BoW code
    foundpose_final.ipynb - is the final checking code

    contrastive_learning_4: is using the model simaese with contrastive loss - not BCE loss

    10* self rotate (not with hardmining) provides the best results

# Content
    Input for cnos should be in dataloader with batch size as
        query_loader is loaded from query_dataset 
            query_dataset is a list of all testing image - zB 150 image
                each element is a dict with keys as (['image', 'scene_id', 'frame_id'])
                    image is (3, H, W) zB ([3, 480, 640])
                        the image is load from its path and is transformed lb
                            self.rgb_transform(image.convert("RGB"))
                    scene_id', 'frame_id of the image zB scene_id = 000001, frame_id =  
                batch_size, H, W, 3 ( which might or might not be normalized - if ja then need to do the inverse before putting thorugh SAM)

        ref_loader is loaded from ref_dataset 
            ref_dataset as a list of (num_templates, 3, 224, 224) # bscailly just all templates (they are /255.0 and then transformed as well)
                For icbin we have only 2 objects - so ref_daset is a list with length 2 only 
    
    output_cnos_analysis_5/train_pbr/cnos_results/detection.npz - npz files are just numpy contains lb
        detection.npz = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
            "segmentation": self.masks,
        }
# Create templates from train_pbr (BlenderProc) or from test (real images)
    In cnos_analysis_4.ipynb - output folder will be in foundpose_analysis folder
    fro real images- use lower templates since we have less images - just 162 are enough I believe
# Create templates from CAD models 
    python -m src.scripts.render_template_with_pyrender level=1 # 0 is for 42 templates, 3 for 2562 templates
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





