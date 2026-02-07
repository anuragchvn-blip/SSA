#!/usr/bin/env python3
"""
Verify that Railway deployment is ready with TLE auto-updater.
Run this before pushing to Railway.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ MISSING {description}: {filepath}")
        return False

def check_file_contains(filepath, search_string, description):
    """Check if a file contains a specific string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ MISSING {description}")
                return False
    except Exception as e:
        print(f"❌ ERROR checking {filepath}: {e}")
        return False

def check_env_vars():
    """Check if required environment variables are set."""
    print("\n=== Environment Variables ===")
    
    required_vars = [
        'SPACETRACK_USERNAME',
        'SPACETRACK_PASSWORD',
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:4] + '***' if len(value) > 4 else '***'
            print(f"✅ {var}: {masked}")
        else:
            print(f"⚠️  {var}: Not set (Railway will need this)")
            all_set = False
    
    return all_set

def main():
    """Main verification."""
    print("=" * 80)
    print("RAILWAY DEPLOYMENT VERIFICATION - TLE AUTO-UPDATER")
    print("=" * 80)
    
    all_checks_passed = True
    
    print("\n=== Core Files ===")
    all_checks_passed &= check_file_exists("requirements.txt", "Requirements file")
    all_checks_passed &= check_file_exists("Procfile", "Railway Procfile")
    all_checks_passed &= check_file_exists("init_database.sh", "Database init script")
    all_checks_passed &= check_file_exists("runtime.txt", "Python runtime specification")
    
    print("\n=== TLE Auto-Updater Files ===")
    all_checks_passed &= check_file_exists("src/data/ingest/tle_auto_updater.py", "TLE Auto-Updater")
    all_checks_passed &= check_file_exists("manage_tle_updater.py", "CLI Management Tool")
    
    print("\n=== Dependencies ===")
    all_checks_passed &= check_file_contains(
        "requirements.txt", 
        "APScheduler", 
        "APScheduler in requirements.txt"
    )
    
    print("\n=== FastAPI Integration ===")
    all_checks_passed &= check_file_contains(
        "src/api/main.py",
        "tle_auto_updater",
        "Auto-updater imported in main.py"
    )
    all_checks_passed &= check_file_contains(
        "src/api/main.py",
        "tle_auto_updater.start()",
        "Auto-updater.start() in lifespan"
    )
    
    print("\n=== Startup Script ===")
    all_checks_passed &= check_file_contains(
        "init_database.sh",
        "uvicorn src.api.main:app",
        "Uvicorn starts FastAPI app"
    )
    all_checks_passed &= check_file_contains(
        "Procfile",
        "init_database.sh",
        "Procfile uses init_database.sh"
    )
    
    # Check environment variables
    check_env_vars()
    
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Ready for Railway deployment!")
        print("\nNext steps:")
        print("1. Ensure SPACETRACK_USERNAME and SPACETRACK_PASSWORD are set in Railway dashboard")
        print("2. Run: git add .")
        print('3. Run: git commit -m "Add 12-hour TLE auto-updater"')
        print("4. Run: git push origin main")
        print("\nRailway will auto-deploy and TLE auto-updater will start immediately.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before deploying")
        return 1

if __name__ == "__main__":
    sys.exit(main())
