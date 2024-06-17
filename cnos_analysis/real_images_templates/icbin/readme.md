# Test on custom image
export OUTPUT_DIR=cnos_analysis/real_images_templates/icbin/obj_000001
export RGB_PATH=datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000001.png

bash cnos_analysis/real_images_templates/icbin/analysis_inference_custom.sh

# For pyrender templates
export OUTPUT_DIR=datasets/bop23_challenge/datasets/templates_pyrender/icbin/obj_000001