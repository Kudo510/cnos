export OUTPUT_DIR=cnos_analysis/real_images_templates/icbin/obj_000001
export RGB_PATH=datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000001.png

python -m cnos_analysis.real_images_templates.icbin.analysis_inference_custom --template_dir $OUTPUT_DIR --rgb_path $RGB_PATH --stability_score_thresh 0.97