import numpy as np

path = "contact_graspnet/results/predictions_11.npz"
data = np.load(path, allow_pickle=True)

print("Keys:", data.files)

# Extract stored Python objects
grasps = data["pred_grasps_cam"].item()
scores = data["scores"].item()
contacts = data["contact_pts"].item()

print("Number of objects with predicted grasps:", len(grasps))
for obj_id in grasps.keys():
    print(f"Object {obj_id}: grasps={grasps[obj_id].shape}, scores={scores[obj_id].shape}, contacts={contacts[obj_id].shape}")
