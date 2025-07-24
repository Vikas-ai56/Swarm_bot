import numpy as np
import heapq

'''
    WORK NEEDED:- 1. Setup the occupancy grid(in `plan_path()`) such that the bots 
                     size is also taken into account to avoid collision.
'''

class Cell:
    def __init__(self):
        self.parent_i = None
        self.parent_j = None
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0  

class HighLevelPlanner:
    def __init__(self, world_size_m, cell_size_m):
        self.grid_width = int(world_size_m[0] / cell_size_m)
        self.grid_height = int(world_size_m[1] / cell_size_m)
        self.cell_size_m = cell_size_m
        print(f"High-level planner initialized with a {self.grid_width}x{self.grid_height} grid.")
        
        self.goal_cell = None
        self.cell_details = [[Cell() for _ in range(self.grid_height)] for _ in range(self.grid_width)]

    def _world_to_grid(self, world_pos):
        grid_x = int(world_pos[0] / self.cell_size_m)
        grid_y = int(world_pos[1] / self.cell_size_m)
        return (grid_x, grid_y)
    
    def _grid_to_world(self, grid_pos):
        world_x = (grid_pos[0] + 0.5) * self.cell_size_m
        world_y = (grid_pos[1] + 0.5) * self.cell_size_m
        return np.array([world_x, world_y])

    def plan_path(self, robot_id, all_poses, goal_id, occupancy_grid):
        """
        --- SOLUTION FOR WORK NEEDED #2 ---
        Plans a global path using A* on an occupancy grid.
        """
        # occupancy_grid = np.zeros((self.grid_height, self.grid_width))

        # OCCUPANCY GRID SETUP
        '''OPTIONAL'''# When setting up another robot as an obstacle we have to keep threshold
        # radius that the robots do not collide at that Point if it can collide only then mark it
        # as an obstacle
        for other_id, other_pose in all_poses.items():
            if robot_id == other_id: continue
            gx, gy = self._world_to_grid(other_pose[:2])
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                occupancy_grid[gy, gx] = 1 # Mark as obstacle

        start_grid = self._world_to_grid(all_poses[robot_id][:2])
        goal_grid = self._world_to_grid(all_poses[goal_id][:2])
        
        path_world_coords = []
        
        node = self.A_star_impl(occupancy_grid,start_grid,goal_grid)
        x , y = all_poses[goal_id][:2]
        path_world_coords.append((x,y))

        while not (x == node.parent_i and y == node.parent_j):
            x = node.parent_i
            y = node.parent_j
# NOTE
#  :- converting back to world Coordinates
# ------------------------------------------------------------------
            path_world_coords.append((self._grid_to_world((x,y))))
# ------------------------------------------------------------------
                   
        path_world_coords.reverse()
        
        return path_world_coords
    
    def A_star_impl(self, occupancy_grid, src, goal):

        visited = dict()

        start = Cell()
        start.f = start.g = 0
        start.parent_i = src[0]
        start.parent_j = src[1]

        visited[(src[0], src[1])] = start

        open_list = []
        heapq.heappush(open_list, (0.0, src[0], src[1]))

        self.cell_details[src[0]][src[1]] = start

        while open_list:
            p = heapq.heappop(open_list)

            i = p[1]
            j = p[2]
            found_goal = False

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            visited[(p[1], p[2])] = True

            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(occupancy_grid, new_i, new_j) and not visited.get((new_i, new_j), False):
                    if self.is_destination(new_i, new_j, goal):
                        curr = Cell()
                        curr.parent_i = i
                        curr.parent_j = j
                        print("Path Planning Successful")
                        self.goal_cell = curr
                        found_goal = True
                        self.cell_details[new_i][new_j] = curr
                        return curr
                    
                    else:
                        g_new = p.g + 1
                        h_new = self.calculate_h_value(new_i, new_j, goal)
                        f_new = g_new + h_new

                        if self.cell_details[new_i][new_j].f == float('inf') or self.cell_details[new_i][new_j].f > f_new:
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            
                            self.cell_details[new_i][new_j].f = f_new
                            self.cell_details[new_i][new_j].g = g_new
                            self.cell_details[new_i][new_j].h = h_new
                            self.cell_details[new_i][new_j].parent_i = i
                            self.cell_details[new_i][new_j].parent_j = j

        if not found_goal:
            print("Could not find goal")

    def is_valid(self, row, col):
        return (row >= 0) and (row < self.grid_width) and (col >= 0) and (col < self.grid_height)

    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 1

    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    def calculate_h_value(self, row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
