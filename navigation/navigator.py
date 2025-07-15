# System Information
CURRENT_UTC_TIME = "2025-05-31 06:18:40"
CURRENT_USER = "mmprateek1"

import numpy as np
import cv2
import time
from queue import Queue

class Navigator:
    def __init__(self):
        self.target_object = None
        self.path_found = False
        self.guidance_queue = Queue()
        self.last_guidance_time = 0
        self.guidance_interval = 1.5
        self.min_safe_distance = 0.3
        self.grid_size = 32
        self.detected_objects = []
        self.last_guidance = None
        self.consecutive_same_guidance = 0
        print("Navigator initialized successfully")

    def set_target(self, object_name):
        self.target_object = object_name.lower()
        self.path_found = False
        print(f"Navigation target set to: {object_name}")
        return f"Setting navigation target to: {object_name}"

    def _find_path(self, grid, start, goal):
        """A* pathfinding implementation"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            y, x = pos
            neighbors = []
            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < grid.shape[0] and 
                    0 <= nx < grid.shape[1] and 
                    grid[ny, nx] > 0.5):  # Using threshold for navigable space
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

    def _get_enhanced_guidance(self, current_pos, next_pos, distance):
        """Enhanced direction guidance for mirrored laptop camera"""
        dy = next_pos[0] - current_pos[0]
        dx = next_pos[1] - current_pos[1]
        
        # Calculate angle for more precise direction
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Distance-based guidance
        if distance < 0.5:
            speed = "very slowly"
            proximity = "You are very close to"
        elif distance < 1.0:
            speed = "slowly"
            proximity = "You are approaching"
        else:
            speed = "steadily"
            proximity = "Moving towards"

        # Direction guidance with clear instructions - reversed for mirrored camera
        if abs(angle) < 15:
            base_direction = f"Keep moving straight ahead {speed}"
        elif angle < -45:
            base_direction = f"Stop. {self.target_object} is to your left. Turn left"
        elif angle < -15:
            base_direction = f"{self.target_object} is slightly left. Turn slightly left"
        elif angle > 45:
            base_direction = f"Stop. {self.target_object} is to your right. Turn right"
        elif angle > 15:
            base_direction = f"{self.target_object} is slightly right. Turn slightly right"
        else:
            base_direction = f"Continue straight ahead {speed}"

        # Add distance context
        if distance < 0.5:
            base_direction += ". Almost there"
        elif distance < 1.0:
            base_direction += ". Getting closer"

        return base_direction

    def get_guidance(self, objects, depth_map):
        if self.target_object is None:
            return {
                'recommendation': "Please specify a target object",
                'path_found': False,
                'distance': None
            }

        # Update detected objects
        self.detected_objects = objects

        # Look for target object
        target_visible = False
        target_distance = None
        target_bbox = None
        
        for obj in objects:
            if obj['class'].lower() == self.target_object:
                target_visible = True
                x1, y1, x2, y2 = obj['bbox']
                target_bbox = obj['bbox']
                
                # Calculate approximate distance using depth map
                target_area = depth_map[y1:y2, x1:x2]
                if target_area.size > 0:
                    target_distance = float(np.mean(target_area))
                break

        if not target_visible:
            self.last_guidance = "Target lost"
            return {
                'recommendation': f"I've lost sight of the {self.target_object}. Please look around slowly.",
                'path_found': False,
                'distance': None
            }

        # Create navigation grid
        height, width = depth_map.shape[:2]
        grid_h, grid_w = height // self.grid_size, width // self.grid_size
        nav_grid = np.ones((grid_h, grid_w), dtype=np.float32)

        # Mark obstacles in grid using depth information
        for y in range(grid_h):
            for x in range(grid_w):
                region = depth_map[y*self.grid_size:(y+1)*self.grid_size, 
                                 x*self.grid_size:(x+1)*self.grid_size]
                if region.size > 0:
                    nav_grid[y, x] = 0 if float(np.mean(region)) < self.min_safe_distance else 1

        # Find target location in grid
        center_x = (target_bbox[0] + target_bbox[2]) // 2
        center_y = (target_bbox[1] + target_bbox[3]) // 2
        target_grid_pos = (center_y // self.grid_size, center_x // self.grid_size)

        # Start position (bottom center of frame)
        start_grid_pos = (grid_h-1, grid_w//2)
        
        # Find path
        path = self._find_path(nav_grid, start_grid_pos, target_grid_pos)
        
        if path is None:
            return {
                'recommendation': f"Cannot find safe path to {self.target_object}.",
                'path_found': False,
                'distance': target_distance
            }

        # Generate guidance
        current_time = time.time()
        if current_time - self.last_guidance_time >= self.guidance_interval:
            if target_distance is not None:
                if target_distance < self.min_safe_distance:
                    guidance = f"Stop! You have reached the {self.target_object}"
                    self.target_object = None
                    self.last_guidance = None
                else:
                    guidance = self._get_enhanced_guidance(path[0], path[1], target_distance)
                    
                    if guidance == self.last_guidance:
                        self.consecutive_same_guidance += 1
                        if self.consecutive_same_guidance > 2:
                            guidance += ". You're following directions well"
                            self.consecutive_same_guidance = 0
                    else:
                        self.consecutive_same_guidance = 0
                    
                    self.last_guidance = guidance
            else:
                guidance = f"Moving towards {self.target_object}. Continue slowly"
            
            self.last_guidance_time = current_time
        else:
            guidance = self.last_guidance if self.last_guidance else f"Following {self.target_object}"

        return {
            'recommendation': guidance,
            'path_found': True,
            'distance': target_distance,
            'path': [(p[1] * self.grid_size + self.grid_size//2, 
                     p[0] * self.grid_size + self.grid_size//2) for p in path],
            'nav_grid': nav_grid
        }

    def visualize_path(self, frame, guidance_result):
        if not guidance_result.get('path_found'):
            return frame

        path = guidance_result.get('path')
        nav_grid = guidance_result.get('nav_grid')
        
        if path and nav_grid is not None:
            # Create path visualization
            path_overlay = np.zeros_like(frame)
            
            # Draw path
            points = np.array(path, dtype=np.int32)
            cv2.polylines(path_overlay, [points], False, (255, 0, 0), 2)
            
            # Draw start and end points
            if len(path) >= 2:
                cv2.circle(path_overlay, tuple(path[0]), 5, (0, 255, 0), -1)  # Start
                cv2.circle(path_overlay, tuple(path[-1]), 5, (0, 0, 255), -1)  # End

            # Combine with original frame
            result = cv2.addWeighted(frame, 1, path_overlay, 0.7, 0)
            return result

        return frame