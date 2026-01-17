"""Demo FastAPI application without database dependencies."""

import os
from contextlib import asynccontextmanager
from typing import List, Optional
import json

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.core.exceptions import BaseSSAException
from src.propagation.sgp4_engine import SGP4Engine
from src.conjunction.probability import ProbabilityCalculator

logger = get_logger(__name__)

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    configure_logging()
    logger.info("SSA Conjunction Analysis Engine Demo started")
    
    yield
    
    # Shutdown
    logger.info("SSA Conjunction Analysis Engine Demo stopped")


# Create FastAPI app
app = FastAPI(
    title="SSA Conjunction Analysis Engine - Demo",
    description="Demonstration of space situational awareness conjunction analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token - simplified for demo."""
    # For demo purposes, accept any token
    return {"user_id": "demo-user", "role": "analyst"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SSA Conjunction Analysis Engine Demo",
        "version": "1.0.0",
        "timestamp": "2026-01-17T07:45:00Z"
    }


# System info endpoint
@app.get("/system/info")
async def system_info(user: dict = Depends(verify_token)):
    """Get system information."""
    return {
        "system_name": "SSA Conjunction Analysis Engine",
        "version": "1.0.0-demo",
        "components": {
            "sgp4_engine": "active",
            "probability_calculator": "active",
            "conjunction_screener": "stub",
            "database": "disabled_demo_mode"
        },
        "capabilities": [
            "TLE parsing and validation",
            "Orbital state propagation",
            "Collision probability calculation",
            "RESTful API interface"
        ]
    }


# Orbital mechanics demo endpoint
@app.get("/demo/orbital-mechanics")
async def orbital_mechanics_demo(user: dict = Depends(verify_token)):
    """Demonstrate orbital mechanics calculations."""
    try:
        engine = SGP4Engine()
        
        # Calculate orbital period for different altitudes
        altitudes_km = [400, 500, 600, 800, 1000]  # km above Earth surface
        earth_radius_km = 6378.137
        results = []
        
        for alt_km in altitudes_km:
            sma_meters = (earth_radius_km + alt_km) * 1000
            period_seconds = engine.calculate_orbital_period(sma_meters)
            period_minutes = period_seconds / 60
            
            results.append({
                "altitude_km": alt_km,
                "semi_major_axis_km": sma_meters / 1000,
                "orbital_period_minutes": round(period_minutes, 2),
                "orbital_velocity_kms": round(2 * 3.14159 * (earth_radius_km + alt_km) / period_minutes, 2)
            })
        
        return {
            "calculation": "orbital_mechanics_demo",
            "results": results,
            "explanation": "Calculated orbital periods for various altitudes using Kepler's third law"
        }
        
    except Exception as e:
        logger.error("Orbital mechanics demo failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Calculation failed")


# Probability calculation demo
@app.get("/demo/probability")
async def probability_demo(user: dict = Depends(verify_token)):
    """Demonstrate collision probability calculations."""
    try:
        calc = ProbabilityCalculator()
        
        # Different scenarios
        scenarios = [
            {
                "name": "Close conjunction",
                "miss_distance": 50.0,  # meters
                "sigma_x": 100.0,       # meters
                "sigma_y": 100.0,       # meters
                "combined_radius": 15.0  # meters
            },
            {
                "name": "Moderate conjunction",
                "miss_distance": 500.0,
                "sigma_x": 150.0,
                "sigma_y": 150.0,
                "combined_radius": 10.0
            },
            {
                "name": "Distant conjunction",
                "miss_distance": 2000.0,
                "sigma_x": 200.0,
                "sigma_y": 200.0,
                "combined_radius": 5.0
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = calc.compute_pc_foster_method(
                miss_distance=scenario["miss_distance"],
                sigma_x=scenario["sigma_x"],
                sigma_y=scenario["sigma_y"],
                combined_radius=scenario["combined_radius"]
            )
            
            results.append({
                "scenario": scenario["name"],
                "miss_distance_m": scenario["miss_distance"],
                "uncertainty_1sigma_m": scenario["sigma_x"],
                "combined_radius_m": scenario["combined_radius"],
                "probability": result.probability,
                "probability_formatted": f"{result.probability:.2e}",
                "risk_level": "HIGH" if result.probability > 1e-3 else "MEDIUM" if result.probability > 1e-6 else "LOW"
            })
        
        return {
            "calculation": "collision_probability_demo",
            "method": "Foster's 2D projection method",
            "scenarios": results,
            "explanation": "Probability of collision calculated using analytical method assuming Gaussian uncertainties"
        }
        
    except Exception as e:
        logger.error("Probability demo failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Probability calculation failed")


# TLE parsing demo
@app.post("/demo/tle-parse")
async def tle_parse_demo(tle_lines: dict, user: dict = Depends(verify_token)):
    """Demonstrate TLE parsing and validation."""
    try:
        line1 = tle_lines.get("line1", "")
        line2 = tle_lines.get("line2", "")
        
        if not line1 or not line2:
            raise HTTPException(status_code=400, detail="Both TLE lines required")
        
        # Basic validation (simplified)
        if len(line1) != 69 or len(line2) != 69:
            raise HTTPException(status_code=400, detail="TLE lines must be 69 characters")
        
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise HTTPException(status_code=400, detail="Invalid TLE format")
        
        # Extract basic information
        norad_id = int(line1[2:7])
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float(f"0.{line2[26:33]}")
        mean_motion = float(line2[52:63])
        
        return {
            "parsed_successfully": True,
            "satellite_info": {
                "norad_id": norad_id,
                "epoch": f"20{epoch_year}-{epoch_day}",
                "orbital_elements": {
                    "inclination_degrees": inclination,
                    "raan_degrees": raan,
                    "eccentricity": eccentricity,
                    "mean_motion_orbits_per_day": mean_motion
                }
            },
            "validation": "basic_syntax_check_passed"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid TLE data: {str(e)}")
    except Exception as e:
        logger.error("TLE parse demo failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="TLE parsing failed")


# Error handlers
@app.exception_handler(BaseSSAException)
async def ssa_exception_handler(request, exc: BaseSSAException):
    """Handle custom SSA exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error_code": exc.error_code.value,
            "message": exc.message,
            "details": exc.details
        }
    )


if __name__ == "__main__":
    # Run development server
    uvicorn.run(
        "demo_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )