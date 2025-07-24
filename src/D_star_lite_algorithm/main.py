# in main.py
import cv2
import numpy as np
import time
import utils

from pose_estimation import PoseEstimation
from planner_layer_2 import PotentialFieldPlanner
from d_star_planner import DStarLitePlanner

# --- Coordinator Configuration ---
ROBOT_ASSIGNMENTS = {5: 6}
CONTROL_LOOP_HZ = 15.0
CONTROL_INTERVAL = 1.0 / CONTROL_LOOP_HZ

def main():
    # --- 1. Initialization ---
    pse = PoseEstimation()
    low_planner = PotentialFieldPlanner()
    
    # Create a dictionary to hold a D* Lite planner for each robot
    planners = {}
    for robot_id in ROBOT_ASSIGNMENTS.keys():
        planners[robot_id] = DStarLitePlanner((utils.REGION_WIDTH, utils.REGION_HEIGHT), 0.1)

    camera_matrix, dist_coeffs = pse.callibrate_init()
    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    # --- State Management ---
    active_assignments = ROBOT_ASSIGNMENTS.copy()
    robot_paths = {}
    last_control_time = 0
    is_initialized = False

    while True:
        ret, frame = video.read()
        if not ret: break

        # --- PERCEPTION LAYER ---
        pse.cv_make_robot_goal_id(frame, utils.ARUCO_DICT["DICT_ARUCO_ORIGINAL"], camera_matrix, dist_coeffs)
        all_poses = pse.cv_fiducial_MarkerDict
        
        # --- CONTROL LOOP (runs at a fixed rate) ---
        current_time = time.time()
        if (current_time - last_control_time) > CONTROL_INTERVAL:
            last_control_time = current_time

            if not all_poses: continue

            # --- One-Time Planner Initialization ---
            if not is_initialized and all(rid in all_poses and gid in all_poses for rid, gid in active_assignments.items()):
                for robot_id, goal_id in active_assignments.items():
                    start_pos = np.array(all_poses[robot_id][:2])
                    goal_pos = np.array(all_poses[goal_id][:2])
                    planners[robot_id].initialize(start_pos, goal_pos)
                is_initialized = True
                print("All planners initialized.")

            if not is_initialized: continue

            # --- DYNAMIC REPLANNING ---
            # Create a set of all current obstacle grid locations
            current_obstacles = set()
            for robot_id in active_assignments.keys():
                if robot_id in all_poses:
                    # Treat OTHER robots as obstacles
                    for other_id in active_assignments.keys():
                        if robot_id != other_id and other_id in all_poses:
                            obs_pos = np.array(all_poses[other_id][:2])
                            current_obstacles.add(planners[robot_id]._world_to_grid(obs_pos))
            
            # --- LOW-LEVEL CONTROL ---
            for robot_id, goal_id in list(active_assignments.items()):
                if robot_id not in all_poses: continue

                # Update the planner with the robot's new position and any new obstacles
                robot_pos = np.array(all_poses[robot_id][:2])
                planners[robot_id].update_and_replan(robot_pos, current_obstacles)
                
                # Get the latest path
                robot_paths[robot_id] = planners[robot_id].get_path()

                if not robot_paths.get(robot_id) or len(robot_paths[robot_id]) < 2:
                    # No path or path is too short, stop the robot
                    # send_command(robot_id, 0, 0)
                    continue

                # The immediate waypoint is the *second* point in the path list
                # (the first is the robot's current location)
                immediate_waypoint = robot_paths[robot_id][1]

                # --- SOLUTION FOR WORK NEEDED #3 ---
                if np.linalg.norm(robot_pos - np.array(all_poses[goal_id][:2])) < low_planner.goal_threshold_m:
                    print(f"!!! SUCCESS: Robot {robot_id} reached goal {goal_id} !!!")
                    del active_assignments[robot_id]
                    continue

                v_cmd, w_cmd = low_planner.get_velocity_commands(robot_id, immediate_waypoint, all_poses)
                v_left, v_right = low_planner.convert_to_wheel_velocities(v_cmd, w_cmd)
                
                print(f"Robot {robot_id} -> Waypoint | Cmds: (v={v_cmd:.2f}, w={w_cmd:.2f})")

        cv2.imshow("D* Lite Coordinator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()