import numpy as np
import cv2
import time
from queue import Queue

# System Information
CURRENT_UTC_TIME = "2025-05-31 04:48:33"
CURRENT_USER = "hk4k"


class PathPlanner:
    def __init__(self):
        self.safety_margin = 5
        self.smoothing_iterations = 5

    def create_traversable_map(self, depth_map):
        depth_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        return (depth_normalized > 0.2).astype(np.uint8)

    def find_path(self, depth_map, start, goal):
        if start is None or goal is None:
            return None, "Invalid start or goal position"
        
        # Create binary traversable map
        traversable = self.create_traversable_map(depth_map)
        
        # Convert points to integer coordinates
        start = (int(start[1]), int(start[0]))
        goal = (int(goal[1]), int(goal[0]))
        
        # Find path using A*
        path = self.astar(traversable, start, goal)
        
        if path is None:
            return None, "No path found"
        
        # Convert back to (x,y) coordinates
        path = [(p[1], p[0]) for p in path]
        return path, "Path found"

    def astar(self, grid, start, goal):
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        
        def get_neighbors(pos):
            y, x = pos
            neighbors = []
            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < grid.shape[0] and 
                    0 <= nx < grid.shape[1] and 
                    grid[ny, nx] > 0):
                    neighbors.append((ny, nx))
            return neighbors

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = frontier.pop(0)[1]
            
            if current == goal:
                break
                
            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    frontier.append((priority, next_pos))
                    frontier.sort()
                    came_from[next_pos] = current

        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        return path



    def create_cost_map(self, depth_map):
        """Create cost map from depth map"""
        # Normalize depth values
        cost_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Invert so that closer objects (smaller depth) have higher cost
        cost_map = 1 - cost_map
        
        # Add safety margin around high-cost areas
        kernel = np.ones((self.safety_margin, self.safety_margin), np.uint8)
        cost_map = cv2.dilate(cost_map, kernel)
        
        return cost_map

    def is_valid_position(self, cost_map, pos):
        """Check if position is within bounds and traversable"""
        y, x = pos
        return (0 <= y < cost_map.shape[0] and 
                0 <= x < cost_map.shape[1] and 
                cost_map[y, x] < 0.8)  # Cost threshold for traversability

    def astar(self, cost_map, start, goal):
        """A* path finding with cost map"""
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

        def get_neighbors(pos):
            y, x = pos
            neighbors = []
            # Include diagonal movements
            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                ny, nx = y + dy, x + dx
                if self.is_valid_position(cost_map, (ny, nx)):
                    neighbors.append((ny, nx))
            return neighbors

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = min(frontier)
            frontier.remove((_, current))
            
            if current == goal:
                break
                
            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + cost_map[next_pos]
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    frontier.append((priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return None

        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        return path

    def smooth_path(self, path):
        """Smooth the path using path relaxation"""
        if not path or len(path) < 3:
            return path

        smoothed = np.array(path, dtype=float)
        
        # Path smoothing weights
        weight_data = 0.5
        weight_smooth = 0.1
        
        for _ in range(self.smoothing_iterations):
            for i in range(1, len(path)-1):
                for j in range(2):  # For both x and y coordinates
                    # Pull toward original path
                    data_factor = weight_data * (path[i][j] - smoothed[i][j])
                    
                    # Pull toward neighboring points
                    smooth_factor = weight_smooth * (
                        smoothed[i-1][j] + smoothed[i+1][j] - 2 * smoothed[i][j]
                    )
                    
                    smoothed[i][j] += data_factor + smooth_factor

        return smoothed.astype(int).tolist()