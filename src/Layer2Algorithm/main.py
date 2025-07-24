import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import math
import time
import utils

from pose_estimation import PoseEstimation
from planner_layer_2 import PotentialFieldPlanner # Or LowLevelPlanner if you rename the file

ROBOT_ASSIGNMENTS = {
    5: 6, 
}
GOAL_THRESHOLD_M = 0.1 

# ---- Rate Controls-----
CONTROL_LOOP_HZ = 15.0 
CONTROL_INTERVAL = 1.0 / CONTROL_LOOP_HZ # Time in seconds between control updates
# Frequency of high-level planner
HIGH_LEVEL_PLANNING_INTERVAL = 3.0 # Re-plan every 3 seconds

def main():
    """The main execution function."""
    
    pse = PoseEstimation()
    immediate_planner = PotentialFieldPlanner(aruco_type = utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"], pse_object = pse)
    
    camera_matrix, dist_coeffs = pse.callibrate_init()
    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

    active_assignments = ROBOT_ASSIGNMENTS.copy()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        pse.cv_make_robot_goal_id(frame,utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"] , camera_matrix, dist_coeffs)
        
        all_poses = pse.cv_fiducial_MarkerDict
        
        if not all_poses:
            cv2.imshow("Multi-Robot Coordinator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue 

        for robot_id, goal_id in list(active_assignments.items()):
            
            if robot_id not in all_poses or goal_id not in all_poses:
                continue

            robot_pos = np.array(all_poses[robot_id][:2])
            goal_pos = np.array(all_poses[goal_id][:2])
            distance_to_goal = np.linalg.norm(robot_pos - goal_pos)

            if distance_to_goal < immediate_planner.goal_threshold:
                print(f"!!! SUCCESS: Robot {robot_id} has reached its goal {goal_id} !!!")
                del active_assignments[robot_id] # Assignment is complete
                continue 

            v_cmd, w_cmd = immediate_planner.get_velocity_commands(robot_id, goal_id)

            v_left, v_right = immediate_planner.convert_to_wheel_velocities(v_cmd, w_cmd)
            
            print(f"Robot {robot_id} -> Goal {goal_id} | Dist: {distance_to_goal:.2f}m | Cmds: (v={v_cmd:.2f}, w={w_cmd:.2f}) | Wheels: (L={v_left:.2f}, R={v_right:.2f})")

        
        cv2.imshow("Multi-Robot Coordinator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()