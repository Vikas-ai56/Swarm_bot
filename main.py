# in main_coordinator.py
import cv2
import numpy as np
import math
import time
import utils

# Import YOUR classes from your files
from pose_estimation import PoseEstimation
from planner_layer_2 import PotentialFieldPlanner # Or LowLevelPlanner if you rename the file

# --- Coordinator Configuration ---

# This solves the core multi-robot problem: WHO goes WHERE.
# This is the "brain" of the operation.
ROBOT_ASSIGNMENTS = {
    5: 6,  # Robot with ID 10 must go to Goal ID 101
    # You can add more assignments here
}
GOAL_THRESHOLD_M = 0.1 # 10 cm tolerance for reaching a goal

def main():
    """The main execution function."""
    
    # 1. Initialize all objects
    pse = PoseEstimation()
    # Note: I am using the corrected LowLevelPlanner class. If you use your
    # original PotentialFieldPlanner, the logic will need to be adapted.
    planner = PotentialFieldPlanner(aruco_type=utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"],pse_object=pse)
    
    # Get camera and calibration data
    camera_matrix, dist_coeffs = pse.callibrate_init()
    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

    # Create a copy of the assignments to safely modify during the loop
    active_assignments = ROBOT_ASSIGNMENTS.copy()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # --- PERCEPTION LAYER ---
        # Call the main update function from your PoseEstimation class
        pse.cv_make_robot_goal_id(frame,utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"] , camera_matrix, dist_coeffs)
        
        # Get the dictionary of all stable, world-frame poses
        all_poses = pse.cv_fiducial_MarkerDict
        
        if not all_poses:
            cv2.imshow("Multi-Robot Coordinator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue # Skip the rest of the loop if no markers are seen

        # --- COORDINATION & CONTROL LAYER ---
        # Loop through the active assignments. Use list() to allow safe deletion.
        for robot_id, goal_id in list(active_assignments.items()):
            
            # Check if both robot and goal are visible
            if robot_id not in all_poses or goal_id not in all_poses:
                continue

            # --- SOLUTION FOR "WORK NEEDED #3" (Goal Reaching Check) ---
            robot_pos = np.array(all_poses[robot_id][:2])
            goal_pos = np.array(all_poses[goal_id][:2])
            distance_to_goal = np.linalg.norm(robot_pos - goal_pos)

            if distance_to_goal < GOAL_THRESHOLD_M:
                print(f"!!! SUCCESS: Robot {robot_id} has reached its goal {goal_id} !!!")
                del active_assignments[robot_id] # Assignment is complete
                # Here you would send a command to stop the robot
                continue # Move to the next assignment

            # --- CONTROL: Get velocity commands from the planner ---
            # Pass all necessary information to the stateless planner method
            v_cmd, w_cmd = planner.get_velocity_commands(robot_id, goal_id)

            # --- SOLUTION FOR "WORK NEEDED #1" (Wheel Velocities) ---
            # This is where you would convert and send the commands
            v_left, v_right = planner.convert_to_wheel_velocities(v_cmd, w_cmd)
            
            print(f"Robot {robot_id} -> Goal {goal_id} | Dist: {distance_to_goal:.2f}m | Cmds: (v={v_cmd:.2f}, w={w_cmd:.2f}) | Wheels: (L={v_left:.2f}, R={v_right:.2f})")
            # your_robot_api.send_speeds(robot_id, v_left, v_right)

        # --- VISUALIZATION ---
        cv2.imshow("Multi-Robot Coordinator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()