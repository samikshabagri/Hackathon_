from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from .simple_working_chatbot import SimpleWorkingChatbot
from .data_loader import DataLoader
from .voyage_planner import VoyagePlanner
from .cargo_matcher import CargoMatcher
from .market_insights import MarketInsights
from .pda_calculator import PDACalculator

# Initialize FastAPI app
app = FastAPI(
    title="Maritime AI Agent",
    description="Intelligent maritime operations assistant with voyage planning, cargo matching, and market analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
chatbot = SimpleWorkingChatbot()
data_loader = DataLoader()
voyage_planner = VoyagePlanner()
cargo_matcher = CargoMatcher()
market_insights = MarketInsights()
pda_calculator = PDACalculator()

# Pydantic models for request validation
class VoyageRequest(BaseModel):
    vessel_imo: str
    load_port: str
    disch_port: str
    speed_knots: Optional[float] = 14.0
    route_variant: Optional[str] = "DIRECT"
    bunker_port: Optional[str] = None

class CargoMatchRequest(BaseModel):
    vessel_imo: str
    min_tce_usd_per_day: Optional[float] = 5000
    max_ballast_distance_nm: Optional[float] = 2000
    cargo_types: Optional[List[str]] = None

class VesselMatchRequest(BaseModel):
    cargo_id: str
    min_tce_usd_per_day: Optional[float] = 5000
    vessel_types: Optional[List[str]] = None

class PDARequest(BaseModel):
    vessel_imo: str
    load_port: str
    disch_port: str
    bunker_port: Optional[str] = None
    fuel_type: Optional[str] = "VLSFO"

class BunkerComparisonRequest(BaseModel):
    vessel_imo: str
    load_port: str
    disch_port: str
    fuel_type: Optional[str] = "VLSFO"
    candidate_ports: List[str]

class MarketAnalysisRequest(BaseModel):
    vessel_type: Optional[str] = None
    route: Optional[str] = None
    days_back: Optional[int] = 30

class ChatRequest(BaseModel):
    query: str

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Maritime AI Agent API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2025-01-27T10:00:00Z"}

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    """Process natural language queries."""
    try:
        result = chatbot.handle_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Voyage planning endpoints
@app.post("/voyage/plan")
async def plan_voyage(request: VoyageRequest):
    """Plan a voyage between two ports."""
    try:
        voyage_plan = voyage_planner.plan_voyage(
            vessel_imo=request.vessel_imo,
            load_port=request.load_port,
            disch_port=request.disch_port,
            speed_knots=request.speed_knots,
            route_variant=request.route_variant,
            bunker_port=request.bunker_port
        )
        return voyage_plan
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/voyage/compare-routes")
async def compare_routes(request: VoyageRequest):
    """Compare different route options for a voyage."""
    try:
        comparisons = voyage_planner.compare_routes(
            vessel_imo=request.vessel_imo,
            load_port=request.load_port,
            disch_port=request.disch_port,
            speeds=[12.0, 14.0, 16.0]
        )
        return {"comparisons": comparisons}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/voyage/optimize-speed")
async def optimize_speed(request: VoyageRequest):
    """Optimize vessel speed for maximum TCE."""
    try:
        optimization = voyage_planner.optimize_speed(
            vessel_imo=request.vessel_imo,
            load_port=request.load_port,
            disch_port=request.disch_port
        )
        return optimization
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Cargo matching endpoints
@app.post("/cargo/find-matches")
async def find_cargo_matches(request: CargoMatchRequest):
    """Find cargo matches for a vessel."""
    try:
        matches = cargo_matcher.find_cargo_matches(
            vessel_imo=request.vessel_imo,
            min_tce_usd_per_day=request.min_tce_usd_per_day,
            max_ballast_distance_nm=request.max_ballast_distance_nm,
            cargo_types=request.cargo_types
        )
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cargo/find-vessel-matches")
async def find_vessel_matches(request: VesselMatchRequest):
    """Find vessel matches for a cargo."""
    try:
        matches = cargo_matcher.find_vessel_matches(
            cargo_id=request.cargo_id,
            min_tce_usd_per_day=request.min_tce_usd_per_day,
            vessel_types=request.vessel_types
        )
        return {"matches": matches}  # Wrap in dict to match frontend expectation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/cargo/optimal-matches")
async def get_optimal_matches(
    min_tce_usd_per_day: float = Query(5000),
    max_matches: int = Query(20)
):
    """Get optimal vessel-cargo combinations."""
    try:
        matches = cargo_matcher.get_optimal_matches(
            min_tce_usd_per_day=min_tce_usd_per_day,
            max_matches=max_matches
        )
        return {"matches": matches}  # Wrap in dict to match frontend expectation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Market analysis endpoints
@app.get("/market/trends/{vessel_type}")
async def get_freight_trends(
    vessel_type: str,
    route: Optional[str] = Query(None)
):
    """Get freight rate trends for a vessel type."""
    try:
        trends = market_insights.get_freight_rate_trends(
            vessel_type=vessel_type,
            route=route
        )
        return trends
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market/summary")
async def get_market_summary():
    """Get overall market summary."""
    try:
        summary = market_insights.get_market_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market/benchmark")
async def benchmark_voyage(
    vessel_imo: str = Query(...),
    load_port: str = Query(...),
    disch_port: str = Query(...)
):
    """Benchmark a voyage against market averages."""
    try:
        benchmark = market_insights.benchmark_voyage_performance(
            vessel_imo=vessel_imo,
            load_port=load_port,
            disch_port=disch_port
        )
        return benchmark
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# PDA calculation endpoints
@app.post("/pda/calculate")
async def calculate_pda(request: PDARequest):
    """Calculate PDA for a voyage."""
    try:
        # First create a voyage plan
        voyage_plan = voyage_planner.plan_voyage(
            vessel_imo=request.vessel_imo,
            load_port=request.load_port,
            disch_port=request.disch_port,
            speed_knots=14.0
        )
        
        # Then calculate PDA using the voyage plan
        pda = pda_calculator.calculate_pda(
            voyage_plan=voyage_plan,
            bunker_port=request.bunker_port,
            fuel_type=request.fuel_type
        )
        return pda
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/pda/compare-bunker-ports")
async def compare_bunker_ports(request: BunkerComparisonRequest):
    """Compare bunker costs across different ports."""
    try:
        # First create a voyage plan
        voyage_plan = voyage_planner.plan_voyage(
            vessel_imo=request.vessel_imo,
            load_port=request.load_port,
            disch_port=request.disch_port,
            speed_knots=14.0
        )
        
        # Then compare bunker ports using the voyage plan
        comparison = pda_calculator.compare_bunker_ports(
            voyage_plan=voyage_plan,
            candidate_ports=request.candidate_ports,
            fuel_type=request.fuel_type
        )
        return {"comparisons": comparison}  # Wrap in dict to match frontend expectation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Data exploration endpoints
@app.get("/data/vessels")
async def get_vessels(
    vessel_type: Optional[str] = Query(None),
    min_dwt: Optional[float] = Query(None),
    max_dwt: Optional[float] = Query(None)
):
    """Get vessel data with optional filtering."""
    try:
        vessels = data_loader.get_vessels(
            vessel_type=vessel_type,
            min_dwt=min_dwt,
            max_dwt=max_dwt
        )
        return {"vessels": vessels}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/cargos")
async def get_cargos(
    cargo_type: Optional[str] = Query(None),
    load_port: Optional[str] = Query(None),
    min_quantity: Optional[float] = Query(None),
    max_quantity: Optional[float] = Query(None)
):
    """Get cargo data with optional filtering."""
    try:
        cargos = data_loader.get_cargos(
            cargo_type=cargo_type,
            load_port=load_port,
            min_quantity=min_quantity,
            max_quantity=max_quantity
        )
        return {"cargos": cargos}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/ports/{port_code}")
async def get_port_info(port_code: str):
    """Get detailed port information."""
    try:
        port_info = data_loader.get_port_info(port_code)
        return port_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Utility endpoints
@app.get("/utils/summary")
async def get_system_summary():
    """Get system summary and statistics."""
    try:
        summary = {
            "vessels_count": len(data_loader.vessels),
            "cargos_count": len(data_loader.cargos),
            "ports_count": len(data_loader.ports),
            "routes_count": len(data_loader.routes),
            "system_status": "operational"
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/utils/commands")
async def get_available_commands():
    """Get list of available commands and examples."""
    try:
        commands = chatbot.get_available_commands()
        return {"commands": commands}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
