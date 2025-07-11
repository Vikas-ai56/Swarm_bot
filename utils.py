import cv2
import math
import time
import numpy as np

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


cameraMatrix = np.array([[3.95728613e+04, 0.00000000e+00, 8.76507235e+02],
				[0.00000000e+00, 3.95633256e+04, 6.19499134e+02],
				[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype='float32')

distCoeffs = np.array([[ 3.20045640e-02,  2.58420867e-03,  2.96728287e-07, -3.09151651e-03, 1.20843467e-06]],dtype='float32')



'''# NOTE:- Make SURE to chenge these according to your map '''

MAP_CORNERS = {4,2,3,0}
REGION_WIDTH = 2
REGION_HEIGHT = 1.5
CV_LOCALIZE_ROBOTS_FIDUCIALS = True
ROBOT_MARKERS = {5}
GOAL_FIDUCIALS = {6}
MARKER_SIZE = 0.06 # in m
# IMPORTANT: Measure this on robot
WHEEL_BASE_M = 0.16  
WHEEL_RADIUS_M = 0.034

def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			# print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image

def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z

def transform_camera_to_world(camera_coords, robot_pose):
    """
    Transform coordinates from camera frame to world frame
    
    Args:
        camera_coords: [x, y, z] in camera coordinate system
        robot_pose: {'x': x, 'y': y, 'z': z, 'yaw': yaw_angle_degrees}
    
    Returns:
        [x, y, z] in world coordinate system
    """
    # Extract camera coordinates
    x_cam, y_cam, z_cam = camera_coords
    
    # Robot pose in world coordinates
    robot_x = robot_pose['x']
    robot_y = robot_pose['y'] 
    robot_z = robot_pose['z']
    robot_yaw = math.radians(robot_pose['yaw'])  # Convert to radians
    
    # Create rotation matrix for robot's yaw angle
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    
    # Apply rotation and translation transformation
    # Assuming camera is mounted facing forward on the robot
    x_world = robot_x + (x_cam * cos_yaw - y_cam * sin_yaw)
    y_world = robot_y + (x_cam * sin_yaw + y_cam * cos_yaw)
    z_world = robot_z + z_cam
    
    return [x_world, y_world, z_world]

def blockingError(errorMsg):
    while(True):
        print(errorMsg)
        time.sleep(1)