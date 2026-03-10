"""
Collision utilities for handling multi-plane environments.
Provides functions to handle collisions with floor and wall planes.
"""

import numpy as np
from typing import List, Optional, Tuple
import json


class PlaneCollisionHandler:
    """Handle collisions with multiple planes."""
    
    def __init__(self, plane_eqs: Optional[List[np.ndarray]] = None):
        """
        Initialize collision handler with plane equations.
        
        Args:
            plane_eqs: List of plane equations [(a, b, c, d), ...] for ax+by+cz+d=0
        """
        self.plane_eqs = plane_eqs if plane_eqs is not None else []
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PlaneCollisionHandler':
        """Load planes from detected environment planes JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        plane_eqs = []
        for plane in data.get('planes', []):
            plane_eqs.append(np.array(plane['equation'], dtype=np.float32))
        
        return cls(plane_eqs)
    
    def signed_distance_to_plane(self, point: np.ndarray, 
                                plane_eq: np.ndarray) -> float:
        """
        Calculate signed distance from point to plane.
        Positive = above plane, Negative = below plane
        """
        a, b, c, d = plane_eq
        norm = np.sqrt(a**2 + b**2 + c**2)
        return (point @ np.array([a, b, c]) + d) / norm
    
    def closest_plane(self, point: np.ndarray) -> Tuple[int, float]:
        """
        Find closest plane and signed distance.
        
        Returns:
            (plane_index, signed_distance)
        """
        min_dist = float('inf')
        closest_idx = -1
        
        for i, plane_eq in enumerate(self.plane_eqs):
            dist = abs(self.signed_distance_to_plane(point, plane_eq))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx, min_dist
    
    def get_plane_normal(self, plane_idx: int) -> np.ndarray:
        """Get normalized normal vector for plane."""
        plane_eq = self.plane_eqs[plane_idx]
        normal = plane_eq[:3]
        return normal / np.linalg.norm(normal)
    
    def get_all_planes_as_arrays(self) -> np.ndarray:
        """Return all plane equations as (N, 4) array."""
        return np.array(self.plane_eqs, dtype=np.float32)


def load_planes_from_config(config_path: str) -> Optional[PlaneCollisionHandler]:
    """Utility to load planes from config file. Returns None if file doesn't exist."""
    try:
        return PlaneCollisionHandler.from_json(config_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def default_ground_plane() -> np.ndarray:
    """Return default ground plane (z = 0)."""
    return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
