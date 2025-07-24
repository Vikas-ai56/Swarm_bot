import heapq
import numpy as np

class DStarLitePlanner:
    def __init__(self, world_size_m, cell_size_m):
        # --- Grid and State Initialization ---
        self.grid_width = int(world_size_m[0] / cell_size_m)
        self.grid_height = int(world_size_m[1] / cell_size_m)
        self.cell_size_m = cell_size_m
        
        # D* Lite state variables
        self.pq = []  # Priority queue (min-heap)
        self.g_scores = {}
        self.rhs_scores = {}
        self.km = 0  # Key modifier for handling path cost changes
        self.start = None
        self.goal = None
        self.obstacles = set()

    # --- Helper functions for grid/world conversion ---
    def _world_to_grid(self, world_pos):
        grid_x = int(world_pos[0] / self.cell_size_m)
        grid_y = int(world_pos[1] / self.cell_size_m)
        return (grid_x, grid_y)

    def _grid_to_world(self, grid_pos):
        world_x = (grid_pos[0] + 0.5) * self.cell_size_m
        world_y = (grid_pos[1] + 0.5) * self.cell_size_m
        return np.array([world_x, world_y])

    # --- Core D* Lite Algorithm Functions ---
    def _heuristic(self, s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2))

    def _get_g(self, s):
        return self.g_scores.get(s, float('inf'))

    def _get_rhs(self, s):
        return self.rhs_scores.get(s, float('inf'))

    def _calculate_key(self, s):
        h = self._heuristic(self.start, s)
        return (min(self._get_g(s), self._get_rhs(s)) + h + self.km, min(self._get_g(s), self._get_rhs(s)))

    def _update_vertex(self, u):
        if u != self.goal:
            min_rhs = float('inf')
            for v in self._get_successors(u):
                min_rhs = min(min_rhs, self._get_g(v) + 1)
            self.rhs_scores[u] = min_rhs
        
               
        if self._get_g(u) != self._get_rhs(u):
            heapq.heappush(self.pq, (self._calculate_key(u), u))

    def _compute_shortest_path(self):
        while self.pq and (heapq.nsmallest(1, self.pq)[0][0] < self._calculate_key(self.start) or self._get_rhs(self.start) != self._get_g(self.start)):
            k_old, u = heapq.heappop(self.pq)
            
            if k_old < self._calculate_key(u):
                heapq.heappush(self.pq, (self._calculate_key(u), u))
                continue
            
            if self._get_g(u) > self._get_rhs(u):
                self.g_scores[u] = self._get_rhs(u)
                # Update the vertices bassed on the new g_score
                for s in self._get_predecessors(u):
                    self._update_vertex(s)
                    
            else:
                self.g_scores[u] = float('inf')
                for s in self._get_predecessors(u):
                    self._update_vertex(s)

    # --- Public API for the Coordinator ---
    def initialize(self, start_m, goal_m):
        self.start = self._world_to_grid(start_m)
        self.goal = self._world_to_grid(goal_m)
        self.rhs_scores[self.goal] = 0
        heapq.heappush(self.pq, (self._calculate_key(self.goal), self.goal))
        self._compute_shortest_path()

    def update_and_replan(self, new_start_m, new_obstacles):
        new_start_grid = self._world_to_grid(new_start_m)
        
        # Update start position and key modifier
        if self.start != new_start_grid:
            self.km += self._heuristic(self.start, new_start_grid)
            self.start = new_start_grid

        # Detect changes in obstacles
        added_obstacles = new_obstacles - self.obstacles
        removed_obstacles = self.obstacles - new_obstacles
        self.obstacles = new_obstacles
        
        # Update vertices affected by new obstacles
        for obs in added_obstacles:
            self.rhs_scores[obs] = float('inf')
            self.g_scores[obs] = float('inf')
            for s in self._get_predecessors(obs):
                self._update_vertex(s)
        
        # Update vertices affected by removed obstacles
        for obs in removed_obstacles:
            self._update_vertex(obs) # Update the node itself first
            for s in self._get_predecessors(obs): # Then update its neighbors
                self._update_vertex(s)

        self._compute_shortest_path()

    def get_path(self):
        if self._get_g(self.start) == float('inf'):
            return None # No path exists

        path = [self._grid_to_world(self.start)]
        current = self.start
        
        while current != self.goal:
            successors = self._get_successors(current)
            if not successors: return None # Stuck

            best_next = min(successors, key=lambda s: self._get_g(s) + 1)
            path.append(self._grid_to_world(best_next))
            current = best_next
        return path

    def _get_successors(self, s):
        successors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = s[0] + dx, s[1] + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in self.obstacles:
                    successors.append((nx, ny))
        return successors

    def _get_predecessors(self, s):
        return self._get_successors(s) # For a grid, they are the same