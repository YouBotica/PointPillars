import open3d as o3d
import numpy as np

def draw_box_o3d(corners, color=[1, 0, 0]):
    # Define the edges of the bounding box
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # lower edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # upper edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    # Create lines set object in open3d
    lines_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # Set the color of the bounding box
    lines_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    return lines_set

def plot_point_cloud_with_bboxes_o3d(point_cloud, annotations):
    # Create open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Assuming the point cloud is (N, 3)

    # Create a list to hold all the geometries to be visualized
    geometries = [pcd]

    # For each car annotation, create and add a bounding box
    for annotation in annotations.get('Car', []):
        h, w, l = annotation['dimensions']
        y, z, x = annotation['location']  # Adjust based on the KITTI dataset's convention

        # Compute x, y, z in velodyne frame
        y = -y
        z = -z

        ry = annotation['rotation_y']

        # Create a bounding box in the vehicle's reference frame
        corners = np.array([
            [l / 2, w / 2, 0],
            [-l / 2, w / 2, 0],
            [-l / 2, -w / 2, 0],
            [l / 2, -w / 2, 0],
            [l / 2, w / 2, h],
            [-l / 2, w / 2, h],
            [-l / 2, -w / 2, h],
            [l / 2, -w / 2, h],
        ])

        # Rotate the bounding box
        rotation_matrix = np.array([
            [np.cos(ry), -np.sin(ry), 0],
            [np.sin(ry), np.cos(ry), 0],
            [0, 0, 1]
        ])
        corners_rotated = np.dot(corners, rotation_matrix.T)

        # Translate the bounding box
        corners_rotated += np.array([x, y, z])

        # Create and add the bounding box to the geometries list
        bbox = draw_box_o3d(corners_rotated)
        geometries.append(bbox)

    # Visualize the point cloud and bounding boxes
    o3d.visualization.draw_geometries(geometries)

    return geometries