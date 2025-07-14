import cv2
from utils import ARUCO_DICT, aruco_display, euler_from_quaternion, MAP_CORNERS
import numpy as np
import math
from camera_calibration import main
import utils

'''
the function "cv_reorder_corners" is not needed if we keep the pallets in order
NOTE:- centerX and centerY are in pixel coordinates. Changes required to convert to meteres. Use rvecs and tvecs
'''

class PoseEstimation:        
    def __init__(self):
        self.cv_fiducial_MarkerDict = dict()
        self.draw_map_corners = True  # Set to False if you don't want to draw the markers on the frame
        self.draw_robot_corners = True  # Set to False if you don't want to draw the robot markers on the frame

    def callibrate_init(self):
        return main()
    
    def cv_get_metric_pose(self, rvec, tvec):
        """Helper function to calculate metric pose from rotation/translation vectors."""
        # Position is the (x, y) from the translation vector
        x_m, y_m, z_m = tvec.flatten()
        
        # Orientation (Yaw)     is calculated from the rotation vector
        rmat, _ = cv2.Rodrigues(rvec)
        yaw_rad = math.atan2(rmat[1, 0], rmat[0, 0])
        
        return (x_m, y_m, yaw_rad)

    def cv_reorder_corners(self):
        # Reorder corners to match the order of MAP_CORNERS
        unsorted_markers = []
        unsorted_markers_Ids = []
        for id in MAP_CORNERS:
            if self.cv_fiducial_MarkerDict.get(id, None) is not None:
                unsorted_markers.append(self.cv_fiducial_MarkerDict[id][0:2])
                unsorted_markers_Ids.append(id)
            else:
                print(self.cv_fiducial_MarkerDict)
                utils.blockingError("Error, Sandbox corner fiducial not found." + str(id))

        top_left = min(unsorted_markers, key=lambda x: x[0] + x[1])
        top_right = max(unsorted_markers, key=lambda x: x[0] - x[1])
        bottom_right = max(unsorted_markers, key=lambda x: x[0] + x[1])
        bottom_left = min(unsorted_markers, key=lambda x: x[0] - x[1])

        top_left_id = unsorted_markers_Ids[unsorted_markers.index(top_left)]
        top_right_id = unsorted_markers_Ids[unsorted_markers.index(top_right)]
        bottom_right_id = unsorted_markers_Ids[unsorted_markers.index(bottom_right)]
        bottom_left_id = unsorted_markers_Ids[unsorted_markers.index(bottom_left)]

        return top_left_id, top_right_id, bottom_left_id, bottom_right_id


    def cv_get_goal_pallets(self):
        goalFiducialIDs = utils.GOAL_FIDUCIALS
        Goal_Ids = []

        # Find and pack the found pallets from the possible pallet fiducials
        for id in goalFiducialIDs:
            if id in self.cv_fiducial_MarkerDict.keys():
                pose = list(self.cv_fiducial_MarkerDict[id][0:2]) + [self.cv_fiducial_MarkerDict[id][6]]
                Goal_Ids.append(pose)
        
        return Goal_Ids
    
    def cv_get_robot_pallets(self):
        goalFiducialIDs = utils.ROBOT_MARKERS
        Robot_pose = []
        Robot_Ids = []

        # Find and pack the found pallets from the possible pallet fiducials
        for id in goalFiducialIDs:
            if id in self.cv_fiducial_MarkerDict.keys():
                pose = list(self.cv_fiducial_MarkerDict[id][0:2]) + [self.cv_fiducial_MarkerDict[id][6]]
                Robot_pose.append(pose)
                Robot_Ids.append(id)
        
        return Robot_pose, Robot_Ids
    
    '''IMPORTANT:- Creates a homogenous transformation matrix {4*4}[Rotation matri(rmat) + translation vector]'''
    def _create_transformation_matrix(self, rvec, tvec):
        mat = np.identity(4)
        rmat, _ = cv2.Rodrigues(rvec)
        mat[0:3, 0:3] = rmat
        mat[0:3, 3] = tvec.flatten()
        return mat

    def _invert_transformation(self, T):
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        R_inv = R.T
        t_inv = -np.dot(R_inv, t)
        T_inv = np.identity(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = t_inv
        return T_inv

    def _get_pose_from_matrix(self, T):
        x = T[0, 3]
        y = T[1, 3]
        yaw = math.atan2(T[1, 0], T[0, 0])
        return (x, y, yaw)

    def cv_make_robot_goal_id(self, frame, type, cameraMatrix=utils.cameraMatrix, distCoeffs=utils.distCoeffs):
        """
        Main detection function. It now performs all steps:
        1. Detect all markers and their camera-relative poses.
        2. Select a map corner as the dynamic origin.
        3. Transform all other marker poses into the origin's frame of reference.
        docs for reference:- https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arucoDict = cv2.aruco.getPredefinedDictionary(type)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected = arucoDetector.detectMarkers(gray)

        # Clear old data before processing the new frame
        self.cv_fiducial_MarkerDict.clear()
        marker_size = utils.MARKER_SIZE

        if ids is None or len(ids) == 0:
            return # No markers detected, nothing to do

        ids = ids.flatten()
        
        # --- STEP 1: Get Camera-Relative Poses for ALL markers using solvePnP ---
        object_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        camera_transforms = {}

        for i, marker_id in enumerate(ids):
            image_points = corners[i]
            
            # Use solvePnP for the current marker
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, utils.cameraMatrix, utils.distCoeffs)
            
            if success:
                # Populate the dictionary with the transform for this marker
                camera_transforms[marker_id] = self._create_transformation_matrix(rvec, tvec)
                # Visualize the camera-relative pose
                cv2.drawFrameAxes(frame, utils.cameraMatrix, utils.distCoeffs, rvec, tvec, marker_size)
                # detected_markers = utils.aruco_display(corners,ids,rejected,frame)

        # --- STEP 2: Dynamically Select the Origin Marker ---
        if not camera_transforms:
            return
        '''Temporary line'''
        # print(camera_transforms.keys())

        origin_id = None
        for map_id in sorted(utils.MAP_CORNERS):
            if map_id in camera_transforms:
                origin_id = map_id
                break
        
        if origin_id is None:
            print("Origin ID not set")
            return

        # --- STEP 3: Calculate the Inverse Transform of the Origin ---
        '''
        In short: The inverse transformation is required to "move" the origin of your coordinate
        system from the camera's lens to the physical ArUco marker you have designated as the 
        world's (0,0,0) point'''
        '''
        THis operation is done so that the origin(landmark for reference) can be set
        to the given marker'''
        T_cam_origin = camera_transforms[origin_id]
        T_origin_cam = self._invert_transformation(T_cam_origin)

        # --- STEP 4: Transform ALL Marker Poses into the World (Origin) Frame ---
        for marker_id, T_cam_marker in camera_transforms.items():
            # T_world = T_world->cam * T_cam->marker  (where T_world->cam is T_origin_cam)
            '''
            Calculations are done from the transformation matrix created w.r.t the origin marker
            not camera's origin (ROBUSTNESS)
            '''
            T_world_marker = np.dot(T_origin_cam, T_cam_marker)
            final_pose = self._get_pose_from_matrix(T_world_marker)
            self.cv_fiducial_MarkerDict[marker_id] = final_pose
        # print(self.cv_fiducial_MarkerDict)
            
        # --- STEP 5: Visualize Markers ---