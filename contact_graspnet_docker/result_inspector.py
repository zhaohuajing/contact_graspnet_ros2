import numpy as np

# # Load the provided .npy files with allow_pickle
# contact_pts = np.load("/mnt/data/contact_pts.npy", allow_pickle=True)
# pred_grasps_cam = np.load("/mnt/data/pred_grasps_cam.npy", allow_pickle=True)
# scores = np.load("/mnt/data/scores.npy", allow_pickle=True)


scene_id = 0
# Load predictions
npz_path = f"contact_graspnet/results/predictions_{scene_id}.npz"
data = np.load(npz_path, allow_pickle=True)
pred_grasps_cam = data["pred_grasps_cam"].item()
scores = data["scores"].item()
contact_pts = data["contact_pts"].item()

def summarize_array(arr, name):
    summary = {
        "name": name,
        "type": type(arr),
    }
    if isinstance(arr, np.ndarray):
        summary.update({
            "dtype": arr.dtype,
            "shape": arr.shape,
        })
        if arr.shape != () and arr.size > 0:
            try:
                summary["element_type"] = type(arr.flat[0])
            except Exception as e:
                summary["element_type"] = f"Error: {e}"
    else:
        summary["info"] = str(arr)
    return summary

summaries = [
    summarize_array(contact_pts, "contact_pts"),
    summarize_array(pred_grasps_cam, "pred_grasps_cam"),
    summarize_array(scores, "scores"),
]


# Since all three loaded arrays are object dtype with shape (),
# they are likely pickled Python objects saved via np.save with allow_pickle=True.
# Let's inspect their actual contents.

contact_pts_obj = contact_pts.items()
pred_grasps_cam_obj = pred_grasps_cam.items()
scores_obj = scores.items()

# Summarize their types and lengths
details = {
    "contact_pts": {"type": type(contact_pts_obj), "len": len(contact_pts_obj) if hasattr(contact_pts_obj, "__len__") else None},
    "pred_grasps_cam": {"type": type(pred_grasps_cam_obj), "len": len(pred_grasps_cam_obj) if hasattr(pred_grasps_cam_obj, "__len__") else None},
    "scores": {"type": type(scores_obj), "len": len(scores_obj) if hasattr(scores_obj, "__len__") else None},
}


# Each file contained a dictionary with 11 entries.
# Let's list the keys and types of their values.

contact_keys = {k: type(v) for k, v in contact_pts_obj}
pred_keys = {k: type(v) for k, v in pred_grasps_cam_obj}
scores_keys = {k: type(v) for k, v in scores_obj}

# print(f'contact_keys = {contact_keys}\n pred_keys = {pred_keys} \n scores_keys = {scores_keys}')

# Let's summarize the shapes and dtypes of the arrays inside each dictionary.
contact_shapes = {k: (v.shape, v.dtype) for k, v in contact_pts_obj}
pred_shapes = {k: (v.shape, v.dtype) for k, v in pred_grasps_cam_obj}
scores_shapes = {k: (v.shape, v.dtype) for k, v in scores_obj}

# print(f'contact_shapes = {contact_shapes} \npred_shapes = {pred_shapes} \nscores_shapes = {scores_shapes}')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

# Plot all keys together, coloring each object's contact points differently
colors = itertools.cycle(plt.cm.tab10.colors)

# # Choose one key with non-empty data,
# for key in range(1,len(contact_pts_obj)+1):
#     pts = contact_pts[float(key)]
#     grasps = pred_grasps_cam[float(key)]
#     scs = scores[float(key)]
#     print(f'object id = {key}, grasp num = {grasps.shape[0]}, scs ={scs}')

#     if len(scs) > 0:
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(pts[:,0], pts[:,1], pts[:,2],  s=10, label="Contact Points") #c='gray',

#         # Plot a few grasp poses (limit to avoid clutter)
#         # num_to_plot = min(10, grasps.shape[0])
#         num_to_plot = grasps.shape[0]
#         for i in range(num_to_plot):
#             T = grasps[i]
#             origin = T[:3, 3]
#             # axes directions scaled by score
#             scale = 0.02 + 0.05 * scs[i]
#             ax.quiver(origin[0], origin[1], origin[2], T[0,0], T[1,0], T[2,0], length=scale, color='r')
#             ax.quiver(origin[0], origin[1], origin[2], T[0,1], T[1,1], T[2,1], length=scale, color='g')
#             ax.quiver(origin[0], origin[1], origin[2], T[0,2], T[1,2], T[2,2], length=scale, color='b')
#         ax.set_title(f"Object {key}")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         # plt.legend()
#         plt.show()

# Plot all keys together, coloring each object's contact points differently
colors = itertools.cycle(plt.cm.tab10.colors)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = itertools.cycle(plt.cm.tab10.colors)
gripper_width = 0.08  # 8 cm typical parallel-jaw gripper width

for key, color in zip(contact_pts.keys(), colors):
    pts = contact_pts[key]
    grasps = pred_grasps_cam[key]
    scs = scores[key]

    if pts.shape[0] > 0 and scs.shape[0] > 0:
        # Plot all contact points faintly
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=5, color=color, alpha=0.3)

        # Best grasp index
        best_idx = np.argmax(scs)
        T = grasps[best_idx]
        origin = T[:3, 3]

        # Closing direction (assume x-axis of grasp frame)
        closing_dir = T[:3, 0]
        half_width = gripper_width / 2.0
        p1 = origin - closing_dir * half_width
        p2 = origin + closing_dir * half_width

        # Highlight antipodal contact points
        ax.scatter(*p1, s=20, color=color, edgecolor='k', marker='o')
        ax.scatter(*p2, s=20, color=color, edgecolor='k', marker='o', label=f"Object {key}")

        # Also plot frame of best grasp
        scale = 0.05
        ax.quiver(origin[0], origin[1], origin[2], T[0,0], T[1,0], T[2,0], length=scale, color='r')
        ax.quiver(origin[0], origin[1], origin[2], T[0,1], T[1,1], T[2,1], length=scale, color='g')
        ax.quiver(origin[0], origin[1], origin[2], T[0,2], T[1,2], T[2,2], length=scale, color='b')

ax.set_title(f"Antipodal Contacts for Best Grasp (Scene {scene_id})")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(markerscale=2, fontsize=8)
plt.show()