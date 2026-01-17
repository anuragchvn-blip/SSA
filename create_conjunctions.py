#!/usr/bin/env python3
"""Create sample conjunction events for testing."""

import datetime
from src.data.database import db_manager
from src.data.storage.conjunction_repository import ConjunctionEventRepository
from src.data.models import ConjunctionEvent

def create_sample_conjunctions():
    """Create sample conjunction events."""
    with db_manager.get_session() as session:
        conj_repo = ConjunctionEventRepository(session)
        
        # Check if we already have conjunction events
        existing_events = conj_repo.get_recent_events(hours_back=1000, limit=100)
        if len(existing_events) > 2:
            print(f"Database already has {len(existing_events)} conjunction events. Skipping creation.")
            return
        
        print("Creating sample conjunction events...")
        
        # Sample conjunction events
        sample_events = [
            {
                "primary_norad_id": 25544,  # ISS
                "secondary_norad_id": 12345,
                "tca_datetime": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2),
                "primary_x_eci": 5111.481,
                "primary_y_eci": -2087.762,
                "primary_z_eci": -3968.625,
                "primary_vx_eci": 4.789,
                "primary_vy_eci": 5.234,
                "primary_vz_eci": -0.892,
                "secondary_x_eci": 5110.234,
                "secondary_y_eci": -2088.987,
                "secondary_z_eci": -3969.123,
                "secondary_vx_eci": 4.785,
                "secondary_vy_eci": 5.231,
                "secondary_vz_eci": -0.895,
                "miss_distance_meters": 850.5,
                "relative_velocity_mps": 7500.0,
                "probability": 0.00023,
                "probability_method": "monte_carlo",
                "probability_confidence_lower": 0.00018,
                "probability_confidence_upper": 0.00028,
                "probability_samples": 10000,
                "screening_threshold_km": 5.0,
                "time_window_hours": 24.0,
                "primary_object_name": "ISS",
                "secondary_object_name": "SAT-12345",
                "primary_object_type": "PAYLOAD",
                "secondary_object_type": "PAYLOAD",
                "primary_radius_meters": 10.0,
                "secondary_radius_meters": 5.0,
                "alert_generated": True,
                "alert_threshold_exceeded": True,
                "alert_sent_at": datetime.datetime.now(datetime.timezone.utc),
                "analysis_version": "1.0.0",
                "algorithm_parameters": {"samples": 10000, "method": "monte_carlo"}
            },
            {
                "primary_norad_id": 44239,  # Starlink
                "secondary_norad_id": 44240,
                "tca_datetime": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=4),
                "primary_x_eci": -1234.317,
                "primary_y_eci": 6083.933,
                "primary_z_eci": -3018.109,
                "primary_vx_eci": -3.456,
                "primary_vy_eci": 2.345,
                "primary_vz_eci": 5.678,
                "secondary_x_eci": -1235.678,
                "secondary_y_eci": 6082.345,
                "secondary_z_eci": -3019.456,
                "secondary_vx_eci": -3.452,
                "secondary_vy_eci": 2.341,
                "secondary_vz_eci": 5.674,
                "miss_distance_meters": 1200.0,
                "relative_velocity_mps": 7200.0,
                "probability": 0.000012,
                "probability_method": "akella",
                "probability_confidence_lower": 0.000008,
                "probability_confidence_upper": 0.000016,
                "probability_samples": 5000,
                "screening_threshold_km": 5.0,
                "time_window_hours": 24.0,
                "primary_object_name": "STARLINK-1234",
                "secondary_object_name": "STARLINK-1235",
                "primary_object_type": "PAYLOAD",
                "secondary_object_type": "PAYLOAD",
                "primary_radius_meters": 2.0,
                "secondary_radius_meters": 2.0,
                "alert_generated": False,
                "alert_threshold_exceeded": False,
                "alert_sent_at": None,
                "analysis_version": "1.0.0",
                "algorithm_parameters": {"samples": 5000, "method": "akella"}
            }
        ]
        
        for i, event_data in enumerate(sample_events):
            try:
                # Create conjunction event
                event = ConjunctionEvent(**event_data)
                created_event = conj_repo.create(event)
                print(f"Created conjunction event {i+1}: {created_event.primary_norad_id} vs {created_event.secondary_norad_id}")
            except Exception as e:
                print(f"Failed to create conjunction event {i}: {e}")
        
        print("Sample conjunction events creation complete!")

if __name__ == "__main__":
    create_sample_conjunctions()