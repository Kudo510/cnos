export LIGHTING_ITENSITY=1.0 # lighting intensity
export RADIUS=0.4 # distance to camera
python -m src.poses.pyrender $CAD_PATH $OUTPUT_DIR False $LIGHTING_ITENSITY $RADIUS