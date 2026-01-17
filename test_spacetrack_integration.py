import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.core.config import settings

async def test_integration():
    print('Testing Space-Track Integration...')
    
    # Check if credentials are configured
    if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
        print('⚠️  Space-Track credentials not configured in environment')
        print('   Please set SPACETRACK_USERNAME and SPACETRACK_PASSWORD in .env')
        return False
    
    username_preview = settings.spacetrack.spacetrack_username[:5] + "..." if settings.spacetrack.spacetrack_username else "None"
    print(f'Username: {username_preview}')
    
    # Initialize client
    rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
    client = SpaceTrackClient(rate_limiter)
    
    try:
        # Test connection
        print('Testing connection...')
        connected = await client.test_connection()
        print(f'Connection: {connected}')
        return connected
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_integration())
    print(f'Result: {result}')