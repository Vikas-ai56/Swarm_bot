# in high_level_planner.py
import numpy as np
# from a_star_library import find_path # A real implementation would be here

class HighLevelPlanner:
    def __init__(self, world_size_m, cell_size_m):
        self.grid_width = int(world_size_m[0] / cell_size_m)
        self.grid_height = int(world_size_m[1] / cell_size_m)
        self.cell_size_m = cell_size_m
        print(f"High-level planner initialized with a {self.grid_width}x{self.grid_height} grid.")

    def _world_to_grid(self, world_pos):
        grid_x = int(world_pos[0] / self.cell_size_m)
        grid_y = int(world_pos[1] / self.cell_size_m)
        return (grid_x, grid_y)
    
    def _grid_to_world(self, grid_pos):
        world_x = (grid_pos[0] + 0.5) * self.cell_size_m
        world_y = (grid_pos[1] + 0.5) * self.cell_size_m
        return np.array([world_x, world_y])

    def plan_path(self, robot_id, all_poses, final_goal_pos):
        """
        --- SOLUTION FOR WORK NEEDED #2 ---
        Plans a global path using A* on an occupancy grid.
        """
        occupancy_grid = np.zeros((self.grid_height, self.grid_width))

        # Mark grid cells occupied by OTHER robots
        for other_id, other_pose in all_poses.items():
            if robot_id == other_id: continue
            gx, gy = self._world_to_grid(other_pose[:2])
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                occupancy_grid[gy, gx] = 1 # Mark as obstacle

        start_grid = self._world_to_grid(all_poses[robot_id][:2])
        goal_grid = self._world_to_grid(final_goal_pos)
        
        # --- A* Placeholder ---
        # A real A* would return a list of grid cells like [(1,1), (1,2), ...]
        # For now, we return a simple line of waypoints.
        path_world_coords = []
        start_world = np.array(all_poses[robot_id][:2])
        for i in range(1, 11):
            waypoint = start_world + (final_goal_pos - start_world) * (i / 10.0)
            path_world_coords.append(waypoint)

        return path_world_coords