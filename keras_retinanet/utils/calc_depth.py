import numpy as np


def calculate_depth(bbox, rotation_matrix, pt_cloud, cp=-1):
    # Find the coordinates of the bbox center point

    rotation_matrix = rotation_matrix.reshape((3,3))
    x1, y1, x2, y2 = bbox
    center_x = (x2 + x1)/2.0
    center_y = (y2 + y1)/2.0

    if cp == -1:
        image_point = np.array([center_x, center_y, 1])
    else:
        image_point = np.array([cp[0], cp[1], 1])

    camera_matrix = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

    depth_plane = 1000 # start guess

    for _ in range(3):
        # project the image point to the 1000 millimeter depth plane.
        temp_cp = np.matmul(np.linalg.inv(camera_matrix), image_point)
        temp_cp = temp_cp * (1.0/temp_cp[2] * depth_plane)


        # Rotate the point cloud and translate
        newPL = np.transpose( np.matmul(rotation_matrix, np.transpose(pt_cloud)) )
        newPL = newPL + temp_cp

        # Project the point cloud into the image plane
        proj_pts = np.matmul(camera_matrix, np.transpose(newPL))
        proj_pts = proj_pts * 1.0/proj_pts[2, :] # Ensure scaling factor is 1

        # Calculate dimension on the new bbox
        x1_proj = np.min([proj_pts[0, :]])
        y1_proj = np.min([proj_pts[1, :]])
        x2_proj = np.max([proj_pts[0, :]])
        y2_proj = np.max([proj_pts[1, :]])

        # calculate difference of the diagonal in the original bbox and the new bbox
        diag_ori = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        diag_new = np.sqrt(np.power((x2_proj - x1_proj), 2) + np.power((y2_proj - y1_proj), 2))
        scale_factor = diag_new/diag_ori

        depth_plane = depth_plane * scale_factor

    return depth_plane
