from flask import Flask
import pathlib
import numpy as np 
import sys 
import tensorflow.compat.v1 as tf
import os 

app = Flask(__name__)


# ugly imports fix
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

from data import  load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image
import config_utils



base_dir = pathlib.Path(__file__).parents[1]

input_npz_path = base_dir / "flask_files" / "flask_input.npz" 
output_npz_path = base_dir / "flask_files" / "flask_output.npz"
output_predictions_visualization_path = base_dir / "flask_files" / "predictions_visualization.png"

ckpt_dir = base_dir / "checkpoints" / "scene_test_2048_bs3_hor_sigma_001"

def build_model():
    global_config = config_utils.load_config(ckpt_dir, batch_size=1)
    print(str(global_config))


     # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, ckpt_dir, mode='test')
    return grasp_estimator, sess


grasp_estimator,tf_session = build_model()

class EstimatorConfig:
    z_range = [0.2, 1.8]
    forward_passes = 1
    local_regions = True
    filter_grasps = True
    skip_border_objects = True


@app.route("/")
def segmap_and_depth_inference():
    input_file = np.load(input_npz_path)
    if not "K" in input_file:
        raise ValueError("K not in input file")
    if not "depth" in input_file:
        raise ValueError("depth not in input file")
    if not "segmap" in input_file:
        raise ValueError("segmap not in input file")
    
    print('loading input data')
    segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(str(input_npz_path))
    if pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                skip_border_objects=EstimatorConfig.skip_border_objects, z_range=EstimatorConfig.z_range)

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(tf_session, pc_full, pc_segments=pc_segments, 
                                                                                        local_regions=EstimatorConfig.local_regions, filter_grasps=EstimatorConfig.filter_grasps, forward_passes=EstimatorConfig.forward_passes)  

    # Save results
    np.savez(str(output_npz_path), pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

    # cannot run Visualization scripts on server so no visualization for now.
    # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=pc_colors, gripper_openings=None, gripper_width=0.08)
    return "inference done"




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")