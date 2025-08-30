import networkx as nx
from geopy.distance import geodesic
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from .data_loader import DataLoader

class VoyagePlanner:
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON-serializable types."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):  # pandas Series
            return obj.to_dict()
        else:
            return obj
    """Handles voyage planning, route optimization, and cost calculations."""
    
    def __init__(self):
        self.data = DataLoader()
        self._build_route_graph()
    
    def _build_route_graph(self):
        """Build a graph of ports with distances for route optimization."""
        self.graph = nx.Graph()
        
        # Add all ports as nodes
        for _, port in self.data.ports.iterrows():
            self.graph.add_node(
                port["code"],
                lat=port["lat"],
                lon=port["lon"],
                country=port["country"],
                region=port["region"]
            )
        
        # Add edges based on existing routes
        for _, route in self.data.routes.iterrows():
            if route["load"] in self.graph.nodes and route["disch"] in self.graph.nodes:
                self.graph.add_edge(
                    route["load"], 
                    route["disch"],
                    distance=route["distance_nm"],
                    variant=route["variant"],
                    canal_toll=route["canal_toll_usd"],
                    canal_delay=route["canal_delay_days"],
                    piracy_risk=route["piracy_risk"],
                    eca_nm=route["eca_nm"]
                )
    
    def calculate_distance(self, port1_code: str, port2_code: str) -> Optional[float]:
        """Calculate great circle distance between two ports in nautical miles."""
        port1 = self.data.get_port_details(port1_code)
        port2 = self.data.get_port_details(port2_code)
        
        if not port1 or not port2:
            return None
        
        coord1 = (port1["lat"], port1["lon"])
        coord2 = (port2["lat"], port2["lon"])
        
        return geodesic(coord1, coord2).nautical
    
    def plan_voyage(self, 
                   vessel_imo: str,
                   load_port: str, 
                   disch_port: str, 
                   speed_knots: float = 14.0,
                   route_variant: str = "DIRECT",
                   bunker_port: Optional[str] = None) -> Dict:
        """
        Plan a complete voyage with costs, ETA, and risks.
        
        Args:
            vessel_imo: Vessel IMO number
            load_port: Loading port code
            disch_port: Discharge port code
            speed_knots: Vessel speed in knots
            route_variant: Route variant (DIRECT, SUEZ, PANAMA, CAPE)
            bunker_port: Optional bunkering port
            
        Returns:
            Dictionary with voyage details
        """
        # Get vessel details
        vessel_imo_str = str(vessel_imo)
        vessel = self.data.vessels[self.data.vessels["imo"] == vessel_imo_str]
        if vessel.empty:
            raise ValueError(f"Vessel {vessel_imo} not found")
        
        vessel_data = vessel.iloc[0]
        
        # Get route information
        route_info = self.data.get_route(load_port, disch_port, route_variant)
        if not route_info:
            # Try to find any route between these ports
            all_routes = self.data.get_all_routes(load_port, disch_port)
            if not all_routes:
                raise ValueError(f"No route found between {load_port} and {disch_port}")
            route_info = all_routes[0]  # Use first available route
        
        # Calculate voyage time
        distance_nm = route_info["distance_nm"]
        voyage_days = distance_nm / (speed_knots * 24)
        
        # Get fuel consumption
        consumption = self.data.get_vessel_consumption(vessel_imo, speed_knots)
        if not consumption:
            raise ValueError(f"Could not get consumption data for vessel {vessel_imo}")
        
        # Calculate fuel costs
        total_fuel_mt = consumption["main_consumption_mt_per_day"] * voyage_days
        
        # Get bunker price
        if bunker_port:
            bunker_price = self.data.get_bunker_price(bunker_port, "VLSFO")
        else:
            # Use load port bunker price
            bunker_price = self.data.get_bunker_price(load_port, "VLSFO")
        
        if not bunker_price:
            bunker_price = 600  # Default price if not available
        
        fuel_cost_usd = total_fuel_mt * bunker_price
        
        # Calculate port costs
        load_port_costs = self.data.get_port_costs(load_port)
        disch_port_costs = self.data.get_port_costs(disch_port)
        
        load_port_total = sum([
            load_port_costs.get("dues_usd", 0),
            load_port_costs.get("pilotage_usd", 0),
            load_port_costs.get("towage_usd", 0),
            load_port_costs.get("agency_usd", 0)
        ]) if load_port_costs else 0
        
        disch_port_total = sum([
            disch_port_costs.get("dues_usd", 0),
            disch_port_costs.get("pilotage_usd", 0),
            disch_port_costs.get("towage_usd", 0),
            disch_port_costs.get("agency_usd", 0)
        ]) if disch_port_costs else 0
        
        # Canal costs
        canal_cost = route_info.get("canal_toll_usd", 0)
        canal_delay_days = route_info.get("canal_delay_days", 0)
        
        # ECA penalty
        eca_nm = route_info.get("eca_nm", 0)
        eca_penalty_cost = 0
        if eca_nm > 0:
            eca_days = eca_nm / (speed_knots * 24)
            eca_penalty_cost = consumption["eca_penalty_mt_per_day"] * eca_days * bunker_price
        
        # Total voyage cost
        total_cost = fuel_cost_usd + load_port_total + disch_port_total + canal_cost + eca_penalty_cost
        
        # Calculate ETA
        total_voyage_days = voyage_days + canal_delay_days
        eta = datetime.now() + timedelta(days=total_voyage_days)
        
        # Risk assessment
        piracy_risk = route_info.get("piracy_risk", "low")
        
        result = {
            "vessel_imo": vessel_imo,
            "vessel_name": vessel_data["name"],
            "vessel_type": vessel_data["type"],
            "dwt": vessel_data["dwt"],
            "load_port": load_port,
            "disch_port": disch_port,
            "route_variant": route_variant,
            "distance_nm": distance_nm,
            "speed_knots": speed_knots,
            "voyage_days": voyage_days,
            "canal_delay_days": canal_delay_days,
            "total_voyage_days": total_voyage_days,
            "eta": eta.isoformat(),
            "fuel_consumption_mt": total_fuel_mt,
            "fuel_cost_usd": fuel_cost_usd,
            "load_port_costs_usd": load_port_total,
            "disch_port_costs_usd": disch_port_total,
            "canal_cost_usd": canal_cost,
            "eca_penalty_cost_usd": eca_penalty_cost,
            "total_voyage_cost_usd": total_cost,
            "bunker_port": bunker_port or load_port,
            "bunker_price_usd_per_mt": bunker_price,
            "piracy_risk": piracy_risk,
            "eca_nm": eca_nm
        }
        
        # Convert numpy types to native Python types
        return self._convert_to_serializable(result)
    
    def compare_routes(self, 
                      vessel_imo: str,
                      load_port: str, 
                      disch_port: str, 
                      speeds: List[float] = [12.0, 14.0, 16.0]) -> List[Dict]:
        """
        Compare different route variants and speeds.
        
        Returns:
            List of voyage plans sorted by total cost
        """
        all_routes = self.data.get_all_routes(load_port, disch_port)
        comparisons = []
        
        for route in all_routes:
            for speed in speeds:
                try:
                    voyage_plan = self.plan_voyage(
                        vessel_imo=vessel_imo,
                        load_port=load_port,
                        disch_port=disch_port,
                        speed_knots=speed,
                        route_variant=route["variant"]
                    )
                    comparisons.append(voyage_plan)
                except Exception as e:
                    print(f"Error planning route {route['variant']} at {speed} knots: {e}")
                    continue
        
        # Sort by total cost
        comparisons.sort(key=lambda x: x["total_voyage_cost_usd"])
        # Convert to serializable format
        return [self._convert_to_serializable(comp) for comp in comparisons]
    
    def optimize_speed(self, 
                      vessel_imo: str,
                      load_port: str, 
                      disch_port: str,
                      route_variant: str = "DIRECT",
                      min_speed: float = 10.0,
                      max_speed: float = 18.0,
                      speed_step: float = 0.5) -> Dict:
        """
        Find the optimal speed for a voyage considering fuel costs vs time.
        
        Returns:
            Dictionary with optimal speed and analysis
        """
        speeds = []
        costs = []
        days = []
        
        for speed in [min_speed + i * speed_step for i in range(int((max_speed - min_speed) / speed_step) + 1)]:
            try:
                voyage = self.plan_voyage(
                    vessel_imo=vessel_imo,
                    load_port=load_port,
                    disch_port=disch_port,
                    speed_knots=speed,
                    route_variant=route_variant
                )
                speeds.append(speed)
                costs.append(voyage["total_voyage_cost_usd"])
                days.append(voyage["total_voyage_days"])
            except Exception as e:
                continue
        
        if not speeds:
            raise ValueError("No valid speed options found")
        
        # Find minimum cost option
        min_cost_idx = costs.index(min(costs))
        min_cost_speed = speeds[min_cost_idx]
        min_cost = costs[min_cost_idx]
        min_cost_days = days[min_cost_idx]
        
        # Find minimum time option
        min_time_idx = days.index(min(days))
        min_time_speed = speeds[min_time_idx]
        min_time_cost = costs[min_time_idx]
        min_time_days = days[min_time_idx]
        
        result = {
            "optimal_speed_knots": min_cost_speed,
            "optimal_cost_usd": min_cost,
            "optimal_days": min_cost_days,
            "fastest_speed_knots": min_time_speed,
            "fastest_cost_usd": min_time_cost,
            "fastest_days": min_time_days,
            "speed_options": list(zip(speeds, costs, days)),
            "cost_savings_fastest": min_time_cost - min_cost,
            "time_savings_optimal": min_cost_days - min_time_days
        }
        return self._convert_to_serializable(result)
    
    def calculate_tce(self, 
                     voyage_plan: Dict, 
                     freight_rate_usd_per_mt: float,
                     cargo_quantity_mt: float) -> Dict:
        """
        Calculate Time Charter Equivalent (TCE) for a voyage.
        
        Args:
            voyage_plan: Voyage plan from plan_voyage()
            freight_rate_usd_per_mt: Freight rate in USD per metric ton
            cargo_quantity_mt: Cargo quantity in metric tons
            
        Returns:
            Dictionary with TCE analysis
        """
        voyage_revenue = freight_rate_usd_per_mt * cargo_quantity_mt
        voyage_cost = voyage_plan["total_voyage_cost_usd"]
        voyage_days = voyage_plan["total_voyage_days"]
        
        # TCE = (Revenue - Costs) / Voyage Days
        tce_usd_per_day = (voyage_revenue - voyage_cost) / voyage_days if voyage_days > 0 else 0
        
        # Calculate ballast costs (simplified - assume 50% of laden voyage cost)
        ballast_cost = voyage_cost * 0.5
        ballast_days = voyage_days * 0.3  # Assume 30% of laden time for ballast
        
        # Round trip TCE
        round_trip_revenue = voyage_revenue
        round_trip_cost = voyage_cost + ballast_cost
        round_trip_days = voyage_days + ballast_days
        round_trip_tce = (round_trip_revenue - round_trip_cost) / round_trip_days if round_trip_days > 0 else 0
        
        result = {
            "voyage_revenue_usd": voyage_revenue,
            "voyage_cost_usd": voyage_cost,
            "voyage_days": voyage_days,
            "tce_usd_per_day": tce_usd_per_day,
            "ballast_cost_usd": ballast_cost,
            "ballast_days": ballast_days,
            "round_trip_revenue_usd": round_trip_revenue,
            "round_trip_cost_usd": round_trip_cost,
            "round_trip_days": round_trip_days,
            "round_trip_tce_usd_per_day": round_trip_tce,
            "profit_margin_percent": ((voyage_revenue - voyage_cost) / voyage_revenue * 100) if voyage_revenue > 0 else 0
        }
        return self._convert_to_serializable(result)
    
    def get_weather_risk(self, route_variant: str) -> Dict:
        """Get weather and piracy risks for a route variant."""
        risk_levels = {
            "DIRECT": {"weather": "low", "piracy": "low"},
            "SUEZ": {"weather": "low", "piracy": "moderate"},
            "PANAMA": {"weather": "low", "piracy": "low"},
            "CAPE": {"weather": "moderate", "piracy": "low"}
        }
        
        return risk_levels.get(route_variant, {"weather": "unknown", "piracy": "unknown"})
    
    def estimate_laytime(self, 
                        port: str, 
                        cargo_type: str, 
                        cargo_quantity_mt: float) -> Dict:
        """
        Estimate laytime for loading/discharging.
        
        Returns:
            Dictionary with laytime estimates
        """
        laytime_rate = self.data.get_laytime_rate(port, cargo_type)
        if not laytime_rate:
            laytime_rate = 1000  # Default rate if not available
        
        laytime_days = cargo_quantity_mt / laytime_rate
        
        result = {
            "port": port,
            "cargo_type": cargo_type,
            "cargo_quantity_mt": cargo_quantity_mt,
            "laytime_rate_mt_per_day": laytime_rate,
            "estimated_laytime_days": laytime_days,
            "estimated_laytime_hours": laytime_days * 24
        }
        return self._convert_to_serializable(result)
