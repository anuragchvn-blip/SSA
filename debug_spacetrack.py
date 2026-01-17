"""
Debug script to test Space-Track API connectivity and understand the error.
"""
import asyncio
import httpx
import json
from datetime import datetime
from urllib.parse import urljoin

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

async def test_basic_auth():
    """Test basic Space-Track authentication."""
    base_url = "https://www.space-track.org/"
    login_endpoint = "ajaxauth/login"
    
    print("Testing Space-Track authentication...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Authenticate first
        auth_data = {
            "identity": settings.spacetrack.spacetrack_username,
            "password": settings.spacetrack.spacetrack_password
        }
        
        try:
            response = await client.post(
                urljoin(base_url, login_endpoint),
                data=auth_data
            )
            
            print(f"Auth Response Status: {response.status_code}")
            print(f"Auth Response Text: {response.text[:500]}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Auth Response JSON: {result}")
                except json.JSONDecodeError:
                    print("Auth response is not JSON")
                
                # Now try a simple query
                print("\nTrying a simple query after authentication...")
                query_url = f"{base_url}basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json"
                
                # Send the same cookies/session as authentication
                response2 = await client.get(query_url)
                
                print(f"Query Response Status: {response2.status_code}")
                print(f"Query Response Text: {response2.text[:1000]}")
                
                if response2.status_code == 200:
                    try:
                        query_result = response2.json()
                        print(f"Query Result: {query_result}")
                    except json.JSONDecodeError as e:
                        print(f"Could not decode query result as JSON: {e}")
                else:
                    print("Query failed!")
            else:
                print("Authentication failed!")
                
        except Exception as e:
            print(f"Error during authentication: {e}")
            import traceback
            traceback.print_exc()

async def main():
    await test_basic_auth()

if __name__ == "__main__":
    asyncio.run(main())