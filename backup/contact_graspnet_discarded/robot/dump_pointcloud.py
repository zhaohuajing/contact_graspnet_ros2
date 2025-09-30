"""script to generate input (.npz file) for contact_graspnet from a zed2i camera. also includes code to manually segment object or region of interest."""

from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_camera_toolkit.utils.annotation_tool import Annotation, get_manual_annotations
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
import numpy as np
import open3d as o3d
import cv2

camera = Zed2i(resolution=Zed2i.RESOLUTION_720,depth_mode=Zed2i.NEURAL_DEPTH_MODE)
camera.runtime_params.texture_confidence_threshold = 97
camera.runtime_params.confidence_threshold = 90

rgb = camera.get_rgb_image()
depth_image = camera.get_depth_image()
cv2.imwrite('depth.png',depth_image)
depth_map = camera.get_depth_map()

intrinsics = camera.intrinsics_matrix()
rgb = camera.get_rgb_image()
rgb = ImageConverter.from_numpy_format(rgb).image_in_opencv_format

import open3d as o3d


rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb),
    o3d.geometry.Image(depth_map),
    depth_scale=1.0,
    depth_trunc=1.0,
    convert_rgb_to_intensity=False,
)

o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    *camera.resolution, intrinsics
)

pcd_legacy = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, o3d_intrinsics
)

o3d.visualization.draw_geometries([pcd_legacy])

pcd = np.asarray(pcd_legacy.points)
# depth map in meters
# segmap is integer indices
segmap = np.zeros_like(depth_map, dtype=np.uint16)


annotation_spec = {
    "polygon": Annotation.Polygon,
}

annotations = get_manual_annotations(rgb, annotation_spec)
polygon = annotations["polygon"]
poly_list = []
for point in polygon:
    poly_list.append(point[0])
    poly_list.append(point[1])
poly_list = [poly_list]
obj_map = BinarySegmentationMask.from_coco_segmentation_mask(poly_list,width=segmap.shape[1],height=segmap.shape[0]).bitmap
segmap[obj_map == 1 ] = 1


# can also provide pc, but that cannot be combined w/ segmap


# use appropriate keys for contact graspnet
np.savez('scene.npz',rgb=rgb,depth=depth_map,K=intrinsics,seg=segmap)


