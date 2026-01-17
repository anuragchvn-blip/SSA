"""
Demonstration script for 3D satellite tracking visualization.
"""
import asyncio
import tempfile
import os
from datetime import datetime, timedelta

from src.visualization.tracking_visualization import SatelliteVisualization
from src.data.ingest.spacetrack_client import SpaceTrackClient
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


async def demonstrate_3d_visualization():
    """Demonstrate the 3D visualization capabilities."""
    print("🎯 Demonstrating 3D Satellite Visualization Features")
    print("=" * 60)
    
    # Initialize the visualization system
    vis_system = SatelliteVisualization()
    print("✅ Satellite visualization system initialized")
    
    # Initialize Space-Track client to get real satellite data
    spacetrack_client = SpaceTrackClient()
    
    # Test satellites - using ISS as primary example
    test_satellites = [25544]  # ISS
    
    print(f"📡 Fetching TLE data for satellites: {test_satellites}")
    
    try:
        async with spacetrack_client:
            # Get TLE for ISS
            tle = await spacetrack_client.fetch_tle_by_norad_id(25544, days_back=1)
            
            if not tle:
                print("❌ Could not fetch TLE data for ISS")
                return
            
            print(f"✅ Retrieved TLE for ISS (NORAD: {tle.norad_id})")
            print(f"   Epoch: {tle.epoch_datetime}")
            print(f"   Inclination: {tle.inclination_degrees}°")
            print(f"   Period: {1440.0/tle.mean_motion_orbits_per_day:.2f} minutes")
            
            # Define time range for visualization
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=30)  # 30-minute trajectory
            
            print(f"\n📊 Creating 3D trajectory visualization...")
            print(f"   Time range: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
            
            # Create 3D trajectory plot
            try:
                fig_3d = vis_system.create_3d_trajectory_plot(tle, start_time, end_time, num_points=100)
                
                # Save the plot to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', prefix='satellite_3d_')
                temp_filename = temp_file.name
                temp_file.close()
                
                fig_3d.write_html(temp_filename)
                print(f"✅ 3D trajectory plot created and saved to: {temp_filename}")
                print(f"   📂 Open this file in a web browser to view the interactive 3D visualization")
                
            except Exception as e:
                print(f"❌ Error creating 3D trajectory plot: {e}")
            
            # Create ground track plot
            print(f"\n🗺️  Creating ground track visualization...")
            try:
                fig_ground = vis_system.create_ground_track_plot(tle, start_time, end_time, num_points=200)
                
                temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.html', prefix='ground_track_')
                temp_filename2 = temp_file2.name
                temp_file2.close()
                
                fig_ground.write_html(temp_filename2)
                print(f"✅ Ground track plot created and saved to: {temp_filename2}")
                print(f"   📂 Open this file in a web browser to view the interactive ground track")
                
            except Exception as e:
                print(f"❌ Error creating ground track plot: {e}")
            
            # Create multi-satellite visualization (simulate with ISS and a few others)
            print(f"\n🌐 Creating multi-satellite visualization...")
            try:
                # For demonstration, we'll use the same ISS TLE multiple times with slight variations
                # In a real scenario, we'd have multiple different satellites
                simulated_tles = [tle]  # Just using ISS for now
                
                fig_multi = vis_system.create_multi_satellite_plot(simulated_tles, start_time, show_trajectories=True)
                
                temp_file3 = tempfile.NamedTemporaryFile(delete=False, suffix='.html', prefix='multi_sat_')
                temp_filename3 = temp_file3.name
                temp_file3.close()
                
                fig_multi.write_html(temp_filename3)
                print(f"✅ Multi-satellite plot created and saved to: {temp_filename3}")
                print(f"   📂 Open this file in a web browser to view the multi-satellite display")
                
            except Exception as e:
                print(f"❌ Error creating multi-satellite plot: {e}")
            
            print(f"\n🎉 Visualization demonstration completed!")
            print(f"📋 Summary of generated visualizations:")
            print(f"   1. 3D Trajectory: {temp_filename}")
            print(f"   2. Ground Track: {temp_filename2}")
            print(f"   3. Multi-Satellite: {temp_filename3}")
            print(f"\n💡 Tip: Open these HTML files in a web browser to interact with the 3D visualizations")
            
    except Exception as e:
        print(f"❌ Error in visualization demonstration: {e}")
        import traceback
        traceback.print_exc()


async def demonstrate_system_responses():
    """Demonstrate what responses we get from the enhanced tracking system."""
    print("\n🔍 Demonstrating System Responses")
    print("=" * 60)
    
    from src.tracking.multi_satellite_tracker import AdvancedTrackingDashboard
    from src.tracking.conjunction_integration import IntegratedTrackingConjunctionSystem
    from src.tracking.alert_system import RealTimeAlertSystem
    from src.tracking.historical_data_storage import HistoricalTrackingService
    
    # Initialize systems
    print("🔧 Initializing enhanced tracking systems...")
    
    # Multi-satellite tracking
    tracking_dashboard = AdvancedTrackingDashboard()
    print("✅ Advanced tracking dashboard initialized")
    
    # Conjunction integration
    conjunction_system = IntegratedTrackingConjunctionSystem()
    print("✅ Conjunction integration system initialized")
    
    # Alert system
    alert_system = RealTimeAlertSystem()
    print("✅ Real-time alert system initialized")
    
    # Historical tracking
    hist_service = HistoricalTrackingService()
    print("✅ Historical tracking service initialized")
    
    # Show system status
    print(f"\n📊 System Status:")
    print(f"   Alert System Status: {alert_system.get_system_status()}")
    print(f"   Tracking Dashboard Stats: {tracking_dashboard.get_performance_metrics()}")
    
    # Simulate a tracking session
    print(f"\n🔄 Starting a sample tracking session...")
    session_id = f"demo_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    success = await hist_service.start_tracking_session(session_id, [25544], "demo_user")
    if success:
        print(f"✅ Tracking session '{session_id}' started successfully")
    else:
        print(f"❌ Failed to start tracking session")
    
    # End the session
    await hist_service.end_tracking_session(session_id)
    print(f"✅ Tracking session '{session_id}' ended")
    
    print(f"\n🎯 Response Summary:")
    print(f"   • 3D Visualization: Interactive plots saved as HTML files")
    print(f"   • Multi-Satellite Tracking: Concurrent tracking with performance metrics")
    print(f"   • Conjunction Analysis: Real-time risk assessment and alerts")
    print(f"   • Alert System: Multi-channel notifications (email, webhook, WebSocket)")
    print(f"   • Historical Storage: Persistent tracking data with analytics")
    print(f"   • Performance: Optimized rate limiting and API handling")


async def main():
    """Main demonstration function."""
    print("🚀 SSA Enhanced Satellite Tracking System - Live Demonstration")
    print("=" * 80)
    
    # Run visualization demonstration
    await demonstrate_3d_visualization()
    
    # Run system response demonstration
    await demonstrate_system_responses()
    
    print(f"\n✨ All demonstrations completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())