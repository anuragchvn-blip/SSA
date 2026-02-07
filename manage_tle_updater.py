#!/usr/bin/env python3
"""
Standalone script to manage and test the TLE auto-updater.

Usage:
  python manage_tle_updater.py run-once    # Run update cycle once
  python manage_tle_updater.py status      # Check status
  python manage_tle_updater.py test        # Test with single satellite (ISS)
"""

import asyncio
import sys
from datetime import datetime, timezone

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.data.ingest.tle_auto_updater import tle_auto_updater
from src.data.database import init_db

logger = get_logger(__name__)


async def run_update_once():
    """Run a single update cycle."""
    print("=" * 80)
    print("TLE Auto-Updater - Single Update Cycle")
    print("=" * 80)
    print()
    
    if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
        print("❌ ERROR: Space-Track credentials not configured!")
        print("   Please set SPACETRACK_USERNAME and SPACETRACK_PASSWORD in .env")
        return False
    
    print(f"✅ Space-Track account: {settings.spacetrack.spacetrack_username}")
    print(f"✅ Rate limit: {settings.spacetrack.spacetrack_rate_limit} requests/hour")
    print()
    
    await tle_auto_updater.update_all_tles()
    
    print()
    print("✅ Update cycle complete!")
    return True


async def test_single_satellite():
    """Test update for a single satellite (ISS)."""
    print("=" * 80)
    print("TLE Auto-Updater - Test Mode (ISS only)")
    print("=" * 80)
    print()
    
    if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
        print("❌ ERROR: Space-Track credentials not configured!")
        return False
    
    print(f"Testing TLE update for ISS (NORAD ID: 25544)")
    print()
    
    success = await tle_auto_updater.update_tle_for_satellite(25544)
    
    if success:
        print("✅ ISS TLE updated successfully!")
    else:
        print("ℹ️  ISS TLE is already up to date or no new data available")
    
    return True


def show_status():
    """Show current updater status."""
    print("=" * 80)
    print("TLE Auto-Updater - Status")
    print("=" * 80)
    print()
    
    if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
        print("❌ Space-Track credentials: NOT CONFIGURED")
        print("   The auto-updater will not run without credentials")
        return
    
    print(f"✅ Space-Track account: {settings.spacetrack.spacetrack_username}")
    print(f"✅ Rate limit: {settings.spacetrack.spacetrack_rate_limit} requests/hour")
    print()
    
    status = tle_auto_updater.get_status()
    
    print(f"Running: {status['is_running']}")
    print(f"Scheduled jobs: {status['scheduled_jobs']}")
    print(f"Next run: {status['next_run']}")
    print()
    print("Rate limiter:")
    print(f"  Requests per hour: {status['rate_limiter']['requests_per_hour']}")
    print(f"  Requests this hour: {status['rate_limiter']['requests_this_hour']}")


def main():
    """Main entry point."""
    configure_logging()
    init_db()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_tle_updater.py run-once    # Run update cycle once")
        print("  python manage_tle_updater.py status      # Check status")
        print("  python manage_tle_updater.py test        # Test with single satellite (ISS)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "run-once":
        asyncio.run(run_update_once())
    elif command == "test":
        asyncio.run(test_single_satellite())
    elif command == "status":
        show_status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
