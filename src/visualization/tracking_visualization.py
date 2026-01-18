"""
Enhanced visualization module for satellite tracking with 3D displays and trajectory plotting.
"""
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from src.data.models import TLE
from src.propagation.sgp4_engine import SGP4Engine
from src.core.logging import get_logger

logger = get_logger(__name__)


class SatelliteVisualization:
    """Enhanced visualization for satellite tracking data."""
    
    def __init__(self):
        self.sgp4_engine = SGP4Engine()
        self.fig = None
        
    def create_3d_trajectory_plot(
        self, 
        tle: TLE, 
        start_time: datetime, 
        end_time: datetime, 
        num_points: int = 100
    ) -> go.Figure:
        """
        Create a 3D plot showing satellite trajectory over time.
        
        Args:
            tle: TLE object for the satellite
            start_time: Start of visualization period
            end_time: End of visualization period
            num_points: Number of points to plot along the trajectory
            
        Returns:
            Plotly figure with 3D trajectory
        """
        # Generate time points
        time_delta = (end_time - start_time).total_seconds()
        time_points = [
            start_time + (end_time - start_time) * i / (num_points - 1)
            for i in range(num_points)
        ]
        
        # Propagate to each time point
        positions = []
        for time_point in time_points:
            try:
                result = self.sgp4_engine.propagate_to_epoch(tle, time_point)
                positions.append([
                    result.cartesian_state.x / 1000,  # Convert to km
                    result.cartesian_state.y / 1000,
                    result.cartesian_state.z / 1000
                ])
            except Exception as e:
                logger.warning(f"Could not propagate to {time_point}: {e}")
                continue
        
        if not positions:
            logger.error("No positions calculated for trajectory plot")
            return go.Figure()
        
        positions = np.array(positions)
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add Earth sphere
        earth_radius_km = 6378.1  # Earth radius in km
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Earth'
        ))
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=3, color='red'),
            name=f'{tle.norad_id} Trajectory'
        ))
        
        # Add starting and ending points
        if len(positions) > 0:
            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                name='Start'
            ))
            fig.add_trace(go.Scatter3d(
                x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='orange', symbol='diamond'),
                name='End'
            ))
        
        fig.update_layout(
            title=f'Satellite {tle.norad_id} Trajectory',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_multi_satellite_plot(
        self, 
        tles: List[TLE], 
        target_time: datetime,
        show_trajectories: bool = False,
        trajectory_duration_hours: float = 1.0
    ) -> go.Figure:
        """
        Create a 3D plot showing multiple satellites at a specific time.
        
        Args:
            tles: List of TLE objects for satellites to display
            target_time: Time at which to evaluate positions
            show_trajectories: Whether to show trajectory paths
            trajectory_duration_hours: Duration to show trajectories in hours
            
        Returns:
            Plotly figure with multiple satellite positions
        """
        fig = go.Figure()
        
        # Add Earth sphere
        earth_radius_km = 6378.1
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Earth'
        ))
        
        # Define colors for different satellites
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        
        for idx, tle in enumerate(tles):
            color = colors[idx % len(colors)]
            
            # Get current position
            try:
                result = self.sgp4_engine.propagate_to_epoch(tle, target_time)
                current_pos = np.array([
                    result.cartesian_state.x / 1000,  # Convert to km
                    result.cartesian_state.y / 1000,
                    result.cartesian_state.z / 1000
                ])
                
                # Add satellite position
                fig.add_trace(go.Scatter3d(
                    x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
                    mode='markers+text',
                    marker=dict(size=8, color=color),
                    text=[f'Sat {tle.norad_id}'],
                    textposition="top center",
                    name=f'Sat {tle.norad_id}'
                ))
                
                # Optionally add trajectory
                if show_trajectories:
                    start_time = target_time
                    from datetime import timedelta
                    end_time = target_time + timedelta(hours=trajectory_duration_hours)
                    
                    # Generate trajectory points
                    num_points = 20
                    time_points = [
                        start_time + (end_time - start_time) * i / (num_points - 1)
                        for i in range(num_points)
                    ]
                    
                    trajectory_positions = []
                    for time_point in time_points:
                        try:
                            traj_result = self.sgp4_engine.propagate_to_epoch(tle, time_point)
                            trajectory_positions.append([
                                traj_result.cartesian_state.x / 1000,
                                traj_result.cartesian_state.y / 1000,
                                traj_result.cartesian_state.z / 1000
                            ])
                        except:
                            continue
                    
                    if trajectory_positions:
                        traj_array = np.array(trajectory_positions)
                        fig.add_trace(go.Scatter3d(
                            x=traj_array[:, 0],
                            y=traj_array[:, 1],
                            z=traj_array[:, 2],
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=f'Traj {tle.norad_id}',
                            showlegend=False
                        ))
                        
            except Exception as e:
                logger.warning(f"Could not propagate satellite {tle.norad_id}: {e}")
                continue
        
        fig.update_layout(
            title='Multi-Satellite Tracking Display',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='cube'
            ),
            width=1000,
            height=700
        )
        
        return fig
    
    def create_ground_track_plot(
        self, 
        tle: TLE, 
        start_time: datetime, 
        end_time: datetime, 
        num_points: int = 200
    ) -> go.Figure:
        """
        Create a ground track plot showing satellite path over Earth's surface.
        
        Args:
            tle: TLE object for the satellite
            start_time: Start of visualization period
            end_time: End of visualization period
            num_points: Number of points to plot along the ground track
            
        Returns:
            Plotly figure with ground track
        """

        
        # Generate time points
        time_points = [
            start_time + (end_time - start_time) * i / (num_points - 1)
            for i in range(num_points)
        ]
        
        # Propagate to each time point and get geographic coordinates
        lats, lons, alts = [], [], []
        for time_point in time_points:
            try:
                result = self.sgp4_engine.propagate_to_epoch(tle, time_point)
                lats.append(result.latitude_deg)
                lons.append(result.longitude_deg)
                alts.append(result.altitude_m / 1000)  # Convert to km
            except Exception as e:
                logger.warning(f"Could not propagate to {time_point}: {e}")
                continue
        
        if not lats:
            logger.error("No ground track points calculated")
            return go.Figure()
        
        # Create ground track plot
        fig = go.Figure()
        
        # Add ground track
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='lines+markers',
            line=dict(width=2, color='red'),
            marker=dict(size=3, color='red'),
            name=f'Sat {tle.norad_id} Ground Track'
        ))
        
        # Add starting point
        if lats and lons:
            fig.add_trace(go.Scattergeo(
                lon=[lons[0]], lat=[lats[0]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond'),
                name='Start'
            ))
            
            fig.add_trace(go.Scattergeo(
                lon=[lons[-1]], lat=[lats[-1]],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='diamond'),
                name='End'
            ))
        
        fig.update_layout(
            title=f'Satellite {tle.norad_id} Ground Track',
            geo=dict(
                showland=True,
                landcolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
                projection_type="orthographic"
            )
        )
        
        return fig


# Import needed for timedelta in the function
import pandas as pd