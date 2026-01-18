# API Documentation

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

## Base URL

`http://localhost:8000` (development)
`https://your-deployment-url.com` (production)

## Endpoints

### Health Check
`GET /health`

Returns the system health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "SSA Conjunction Analysis Engine",
  "version": "1.0.0",
  "timestamp": "2026-01-17T14:30:00.000000Z"
}
```

### System Status
`GET /status`

Retrieves current system status and statistics.

**Response:**
```json
{
  "system_status": "operational",
  "database_connected": true,
  "active_satellites": 150,
  "tle_records": 1200,
  "conjunctions_today": 5,
  "high_risk_conjunctions_today": 0,
  "last_data_ingestion": "2026-01-17T13:45:00.000000Z",
  "uptime_minutes": 1440
}
```

### Satellite Positions
`GET /satellites/positions`

Retrieves real-time satellite positions with propagated orbits.

**Response:**
```json
{
  "timestamp": "2026-01-17T14:30:00.000000Z",
  "satellites": [
    {
      "norad_id": 25544,
      "name": "ISS",
      "x": -6299.93,
      "y": -743.56,
      "z": 2441.41,
      "lat": 21.05,
      "lon": 40.04,
      "alt": 419.10,
      "risk": "nominal"
    }
  ]
}
```

### Satellite Catalog
`GET /satellites/catalog`

Retrieves the full satellite catalog with metadata.

**Response:**
```json
{
  "count": 150,
  "satellites": [
    {
      "norad_id": 25544,
      "name": "SAT-25544",
      "type": "PAYLOAD",
      "lat": 0.0,
      "lon": 0.0,
      "alt": 0.0,
      "velocity": 7500.0,
      "status": "active"
    }
  ]
}
```

### Institutional Catalog
`GET /satellites/institutional-catalog`

Retrieves the institutional satellite catalog with advanced filtering.

**Parameters:**
- `type_filter` (optional): Filter by satellite type (PAYLOAD, ROCKET BODY, DEBRIS)
- `search` (optional): Search by NORAD ID or name
- `limit` (optional): Maximum number of results (default: 100)

**Response:**
```json
{
  "count": 150,
  "satellites": [
    {
      "norad_id": 25544,
      "common_name": "SAT-25544",
      "type": "PAYLOAD",
      "inclination_deg": 51.64,
      "apogee_km": 400.0,
      "perigee_km": 400.0,
      "period_minutes": 92.4,
      "rcs_m2": 12.5,
      "status": "ACTIVE"
    }
  ]
}
```

### Conjunction Screening
`POST /conjunctions/screen`

Screen for potential conjunctions with a specified satellite.

**Request Body:**
```json
{
  "primary_norad_id": 25544,
  "time_window_hours": 24.0,
  "screening_threshold_km": 5.0,
  "probability_threshold": 1e-6,
  "include_debris": false
}
```

**Response:**
```json
{
  "primary_norad_id": 25544,
  "screening_complete": true,
  "total_candidates": 15,
  "conjunctions_found": 2,
  "high_risk_conjunctions": 0,
  "events": [
    {
      "secondary_norad_id": 12345,
      "tca_datetime": "2026-01-18T15:30:00.000000Z",
      "miss_distance_meters": 1250.0,
      "probability": 1.2e-7,
      "relative_velocity_mps": 7500.0,
      "risk_level": "low"
    }
  ],
  "analysis_duration_ms": 125.4,
  "processing_details": {
    "initial_screening_time_ms": 45.2,
    "refinement_time_ms": 78.1,
    "covariance_time_ms": 2.1
  }
}
```

### Conjunction Events
`GET /conjunctions/events`

Retrieve recent conjunction events.

**Parameters:**
- `hours_back` (optional): Hours back to query (default: 24)
- `min_probability` (optional): Minimum probability threshold (default: 1e-6)
- `limit` (optional): Maximum number of results (default: 50)

**Response:**
```json
{
  "events": [
    {
      "id": 1,
      "primary_norad_id": 25544,
      "secondary_norad_id": 12345,
      "tca_datetime": "2026-01-17T15:30:00.000000Z",
      "miss_distance_meters": 850.5,
      "probability": 0.00023,
      "relative_velocity_mps": 7500.0,
      "risk_level": "high"
    }
  ],
  "count": 2,
  "parameters": {
    "hours_back": 24,
    "min_probability": 1e-6,
    "limit": 50
  }
}
```

### Recent Alerts
`GET /alerts/recent`

Retrieve recent system alerts.

**Parameters:**
- `limit` (optional): Maximum number of results (default: 50)

**Response:**
```json
{
  "alerts": [
    {
      "id": 1,
      "alert_type": "CONJUNCTION",
      "primary_norad_id": 25544,
      "secondary_norad_id": 12345,
      "severity": "HIGH",
      "probability": 0.00023,
      "miss_distance_meters": 850.5,
      "tca_datetime": "2026-01-17T15:30:00.000000Z",
      "generated_at": "2026-01-17T14:25:00.000000Z",
      "status": "OPEN"
    }
  ],
  "count": 1,
  "last_updated": "2026-01-17T14:30:00.000000Z"
}
```

### Intelligence Summary
`GET /intelligence/summary`

Get intelligence and security summary.

**Response:**
```json
{
  "active_threats": 1,
  "conjunction_events_24h": 15,
  "high_risk_events": 1,
  "maneuver_detections": 3,
  "tracked_objects": 150,
  "system_integrity": "100%",
  "global_coverage": "94.2%",
  "threat_level": "ELEVATED"
}
```

### Priority Targets
`GET /intelligence/priority-targets`

Get priority targets for monitoring.

**Response:**
```json
{
  "targets": [
    {
      "id": "OBJ-25544",
      "norad_id": 25544,
      "name": "OBJ-25544",
      "interest_level": "High Interest",
      "last_contact": "14:30:15 Z",
      "orbit": "LEO / 51.6°",
      "risk_score": 21.05
    }
  ],
  "count": 5
}
```

### Conjunctions Summary
`GET /intelligence/conjunctions-summary`

Get conjunction events summary.

**Response:**
```json
{
  "active_conjunctions": 1,
  "total_events_24h": 15,
  "high_risk_events": 1,
  "events": [
    {
      "id": 1,
      "primary_norad_id": 25544,
      "secondary_norad_id": 12345,
      "tca_datetime": "2026-01-17T15:30:00.000000Z",
      "miss_distance_meters": 850.5,
      "probability": 0.00023,
      "relative_velocity_mps": 7500.0
    }
  ]
}
```

### Latest TLE
`GET /tle/latest/{norad_id}`

Get the latest TLE for a specific satellite.

**Response:**
```json
{
  "id": 1,
  "norad_id": 25544,
  "tle_line1": "1 25544U 98067A   26017.59740489  .00016717  00000-0  25000-3 0  9996",
  "tle_line2": "2 25544  51.6437 208.9163 0001758 353.1409 212.8506 15.49573593 67890",
  "epoch_datetime": "2026-01-17T14:20:00.000000Z",
  "created_at": "2026-01-17T14:20:00.000000Z",
  "is_valid": true
}
```

### Maneuver Detection
`POST /maneuvers/detect`

Detect potential maneuvers for a satellite.

**Parameters:**
- `norad_id`: Satellite NORAD ID
- `days_back`: Days to look back for maneuver detection (default: 7)

**Response:**
```json
{
  "norad_id": 25544,
  "analysis_period_days": 7,
  "maneuvers_detected": 0,
  "confidence": 0.95,
  "last_stable_epoch": "2026-01-10T12:00:00.000000Z",
  "anomalies": [],
  "detection_method": "statistical_orbital_deviation"
}
```