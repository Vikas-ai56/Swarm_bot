
import cv2
import numpy as np
import math
import time
import utils

from pose_estimation import PoseEstimation
from planner_layer_2 import PotentialFieldPlanner # Or LowLevelPlanner if you rename the file
from planner_layer_1 import HighLevelPlanner

ROBOT_ASSIGNMENTS = {
    5: 6, 
}
GOAL_THRESHOLD_M = 0.1 
planning_order = ROBOT_ASSIGNMENTS.keys()

# ---- Rate Controls-----
CONTROL_LOOP_HZ = 5.0 
CONTROL_INTERVAL = 1.0 / CONTROL_LOOP_HZ # Time in seconds between control updates
# Frequency of high-level planner
HIGH_LEVEL_PLANNING_INTERVAL = 3.0 # Re-plan every 3 seconds
last_plan_time = last_control_time = float('-inf')

def main():
    """The main execution function."""
    
    pse = PoseEstimation()
    path_planner = HighLevelPlanner((utils.REGION_WIDTH,utils.REGION_HEIGHT), 0.1) # each cell in the occupancy grid has side_length = 10cm
    immediate_planner = PotentialFieldPlanner(aruco_type = utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"], pse_object = pse)
    robot_paths = {}
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

        # Sending velocity/control commands
        if time.time() - last_control_time >= CONTROL_INTERVAL:
            last_control_time = time.time()

            if time.time() - last_plan_time >= HIGH_LEVEL_PLANNING_INTERVAL:
                master_occupancy_grid = np.zeros((path_planner.grid_height, path_planner.grid_width))

                # Sequential Path planning
                for robotId in planning_order:
                    if robotId in active_assignments:
                        if robot_id in all_poses and goal_id in all_poses:

                            path = path_planner.plan_path(robotId,all_poses,goal_id,master_occupancy_grid)
                            robot_paths[robotId] = path
                            
                # In this loop even the goal pallet of any robot is considered as obstacle for another robot
                            for waypoint in path:
                                gx, gy = path_planner._world_to_grid(waypoint)
                                if 0 <= gx < path_planner.grid_width and 0 <= gy < path_planner.grid_height:
                                    master_occupancy_grid[gy, gx] = 1

                last_plan_time = time.time()

            for robot_id, goal_id in list(active_assignments.items()):
                if robot_id not in all_poses or robot_id not in robot_paths or not robot_paths[robot_id]:
                    continue

                immediate_waypoint = robot_paths[robot_id][0]

                robot_pos = np.array(all_poses[robot_id][:2])
                if np.linalg.norm(robot_pos - immediate_waypoint) < immediate_planner.goal_threshold:
                    robot_paths[robot_id].pop(0) # Waypoint reached, remove it
                    
                    # Check if the final goal is reached (path is now empty)
                    if not robot_paths[robot_id]:
                        print(f"!!! SUCCESS: Robot {robot_id} has reached its final goal {goal_id} !!!")
                        del active_assignments[robot_id]
                        continue 

                v_cmd, w_cmd = immediate_planner.get_velocity_commands(robot_id, goal_id, immediate_waypoint)

                v_left, v_right = immediate_planner.convert_to_wheel_velocities(v_cmd, w_cmd)
                
                print(f"Robot {robot_id} -> Waypoint | Cmds: (v={v_cmd:.2f}, w={w_cmd:.2f}) | Wheels: (L={v_left:.2f}, R={v_right:.2f})")
        
        cv2.imshow("Multi-Robot Coordinator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

'''
import requests
import json
import time

# --- Robot Communication Configuration ---
# You must find the IP address of your robot first.
# You can program the ESP8266 to print its IP to the Serial Monitor on startup.
ROBOT_IP_ADDRESS = "192.168.1.105" # Example IP address
ROBOT_URL = f"http://{ROBOT_IP_ADDRESS}/set_velocity"
REQUEST_TIMEOUT = 0.5 # Seconds

def send_command_to_robot(v_left, v_right):
    """
    Sends wheel velocity commands to the robot via an HTTP POST request.
    """
    try:
        # 1. Format the data into a JSON payload
        payload = {"left": v_left, "right": v_right}
        
        # 2. Send the POST request
        response = requests.post(ROBOT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        
        # 3. Check for success (optional but good practice)
        if response.status_code == 200:
            # print(f"Successfully sent command: {payload}")
            return True
        else:
            print(f"Error sending command: Received status code {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        # This catches network errors like timeouts or connection failures
        print(f"Network error sending command: {e}")
        return False

# In your main loop, you would call this function:
# v_left, v_right = low_planner.convert_to_wheel_velocities(v_cmd, w_cmd)
# send_command_to_robot(v_left, v_right)
'''