"""Demo script showing all SSA engine capabilities."""

import requests
import json

BASE_URL = "http://localhost:8000"
TEST_TOKEN = "test-token"  # Matches the demo auth check

def test_health_endpoint():
    """Test health check endpoint."""
    print("=== HEALTH CHECK ===")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()

def test_system_info():
    """Test system information endpoint."""
    print("=== SYSTEM INFORMATION ===")
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    response = requests.get(f"{BASE_URL}/system/info", headers=headers)
    print(json.dumps(response.json(), indent=2))
    print()

def test_orbital_mechanics():
    """Test orbital mechanics calculations."""
    print("=== ORBITAL MECHANICS DEMO ===")
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    response = requests.get(f"{BASE_URL}/demo/orbital-mechanics", headers=headers)
    result = response.json()
    print(f"Calculation: {result['calculation']}")
    print("Results:")
    for res in result['results']:
        print(f"  Altitude {res['altitude_km']}km: {res['orbital_period_minutes']} minutes per orbit")
    print()

def test_probability_calculations():
    """Test collision probability calculations."""
    print("=== COLLISION PROBABILITY DEMO ===")
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    response = requests.get(f"{BASE_URL}/demo/probability", headers=headers)
    result = response.json()
    print(f"Method: {result['method']}")
    print("Scenarios:")
    for scenario in result['scenarios']:
        print(f"  {scenario['scenario']}:")
        print(f"    Miss distance: {scenario['miss_distance_m']}m")
        print(f"    Probability: {scenario['probability_formatted']}")
        print(f"    Risk level: {scenario['risk_level']}")
    print()

def test_tle_parsing():
    """Test TLE parsing functionality."""
    print("=== TLE PARSING DEMO ===")
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    # Sample ISS TLE
    tle_data = {
        "line1": "1 25544U 98067A   26017.50000000  .00010000  00000-0  10000-3 0  1234",
        "line2": "2 25544  51.6416 123.4567 0001234  45.6789 123.4567 15.50000000123450"
    }
    response = requests.post(f"{BASE_URL}/demo/tle-parse", headers=headers, json=tle_data)
    result = response.json()
    print(f"Parsing successful: {result['parsed_successfully']}")
    print(f"Satellite NORAD ID: {result['satellite_info']['norad_id']}")
    print(f"Inclination: {result['satellite_info']['orbital_elements']['inclination_degrees']}°")
    print()

def main():
    """Run all demo tests."""
    print("🚀 SSA Conjunction Analysis Engine - Live Demo")
    print("=" * 50)
    
    try:
        test_health_endpoint()
        test_system_info()
        test_orbital_mechanics()
        test_probability_calculations()
        test_tle_parsing()
        
        print("✅ All demonstrations completed successfully!")
        print("\n🔧 Server is running at http://localhost:8000")
        print("📚 API documentation available at http://localhost:8000/api/docs")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the server is running:")
        print("   python demo_app.py")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()