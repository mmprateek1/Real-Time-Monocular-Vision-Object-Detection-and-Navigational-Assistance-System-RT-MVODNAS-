import numpy as np
import cv2

class EnvironmentDescriptor:
    def __init__(self):
        self.object_categories = {
            'furniture': ['chair', 'couch', 'bed', 'table', 'desk'],
            'electronics': ['tv', 'laptop', 'monitor', 'phone'],
            'obstacles': ['person', 'dog', 'cat', 'potted plant'],
            'structure': ['door', 'window', 'wall']
        }
        
    def analyze(self, frame, objects, depth_map):
        height, width = frame.shape[:2]
        
        # Categorize detected objects
        categorized_objects = {category: [] for category in self.object_categories}
        for obj in objects:
            for category, items in self.object_categories.items():
                if obj['class'] in items:
                    categorized_objects[category].append(obj)
        
        # Analyze spatial relationships
        spatial_info = self._analyze_spatial_relationships(objects, depth_map, width)
        
        # Generate environment description
        description = self._generate_description(categorized_objects, spatial_info)
        
        return {
            'categorized_objects': categorized_objects,
            'spatial_info': spatial_info,
            'description': description
        }
    
    def _analyze_spatial_relationships(self, objects, depth_map, width):
        relationships = []
        
        for i, obj1 in enumerate(objects):
            x1_center = (obj1['bbox'][0] + obj1['bbox'][2]) // 2
            
            # Determine position (left, center, right)
            position = 'center'
            if x1_center < width//3:
                position = 'left'
            elif x1_center > 2*width//3:
                position = 'right'
                
            relationships.append({
                'object': obj1['class'],
                'position': position
            })
        
        return relationships
    
    def _generate_description(self, categorized_objects, spatial_info):
        description = []
        
        # Add furniture information
        if categorized_objects['furniture']:
            furniture = [obj['class'] for obj in categorized_objects['furniture']]
            description.append(f"Furniture detected: {', '.join(furniture)}")
        
        # Add obstacle information
        if categorized_objects['obstacles']:
            obstacles = [obj['class'] for obj in categorized_objects['obstacles']]
            description.append(f"Potential obstacles: {', '.join(obstacles)}")
        
        # Add spatial information
        for rel in spatial_info:
            description.append(f"{rel['object']} is in the {rel['position']} area")
        
        return ' '.join(description)