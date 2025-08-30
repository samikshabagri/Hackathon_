from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .data_loader import DataLoader
from .voyage_planner import VoyagePlanner

class CargoMatcher:
    """Handles vessel-cargo matching and profitability analysis."""
    
    def __init__(self):
        self.data = DataLoader()
        self.voyage_planner = VoyagePlanner()
    
    def find_cargo_matches(self, 
                          vessel_imo: str,
                          min_tce_usd_per_day: float = 5000,
                          max_ballast_distance_nm: float = 2000,
                          cargo_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Find suitable cargoes for a specific vessel.
        
        Args:
            vessel_imo: Vessel IMO number
            min_tce_usd_per_day: Minimum TCE required
            max_ballast_distance_nm: Maximum ballast distance
            cargo_types: List of preferred cargo types
            
        Returns:
            List of cargo matches sorted by TCE
        """
        # Get vessel details
        vessel = self.data.vessels[self.data.vessels["imo"] == vessel_imo]
        if vessel.empty:
            raise ValueError(f"Vessel {vessel_imo} not found")
        
        vessel_data = vessel.iloc[0]
        vessel_dwt = vessel_data["dwt"]
        vessel_type = vessel_data["type"]
        
        # Filter cargoes by DWT compatibility (±10%)
        min_cargo_qty = vessel_dwt * 0.9
        max_cargo_qty = vessel_dwt * 1.1
        
        # Get suitable cargoes
        cargoes = self.data.get_cargos(
            min_quantity=min_cargo_qty,
            max_quantity=max_cargo_qty
        )
        
        if cargo_types:
            cargoes = [c for c in cargoes if c["commodity"] in cargo_types]
        
        matches = []
        
        for cargo in cargoes:
            try:
                # Plan voyage for this cargo
                voyage_plan = self.voyage_planner.plan_voyage(
                    vessel_imo=vessel_imo,
                    load_port=cargo["load_port"],
                    disch_port=cargo["disch_port"],
                    speed_knots=14.0  # Default speed
                )
                
                # Get freight rate estimate
                route_key = f"{cargo['load_port']}-{cargo['disch_port']}"
                freight_rate = self.data.get_freight_rate_estimate(vessel_type, route_key)
                if not freight_rate:
                    freight_rate = 30.0  # Default rate
                
                # Calculate TCE
                tce_analysis = self.voyage_planner.calculate_tce(
                    voyage_plan=voyage_plan,
                    freight_rate_usd_per_mt=freight_rate,
                    cargo_quantity_mt=cargo["qty_mt"]
                )
                
                # Check if TCE meets minimum requirement
                if tce_analysis["tce_usd_per_day"] >= min_tce_usd_per_day:
                    # Estimate ballast distance (simplified)
                    ballast_distance = self._estimate_ballast_distance(
                        cargo["load_port"], 
                        max_ballast_distance_nm
                    )
                    
                    if ballast_distance <= max_ballast_distance_nm:
                        match = {
                            "cargo_id": cargo["cargo_id"],
                            "commodity": cargo["commodity"],
                            "quantity_mt": cargo["qty_mt"],
                            "load_port": cargo["load_port"],
                            "disch_port": cargo["disch_port"],
                            "laycan_open": cargo["laycan_open"].isoformat() if hasattr(cargo["laycan_open"], 'isoformat') else str(cargo["laycan_open"]),
                            "laycan_close": cargo["laycan_close"].isoformat() if hasattr(cargo["laycan_close"], 'isoformat') else str(cargo["laycan_close"]),
                            "stowage_factor": cargo["stowage_factor_m3_per_mt"],
                            "terms": cargo["terms"],
                            "voyage_plan": voyage_plan,
                            "freight_rate_usd_per_mt": freight_rate,
                            "tce_analysis": tce_analysis,
                            "ballast_distance_nm": ballast_distance,
                            "vessel_compatibility_score": self._calculate_compatibility_score(vessel_data, cargo)
                        }
                        matches.append(match)
                
            except Exception as e:
                print(f"Error processing cargo {cargo['cargo_id']}: {e}")
                continue
        
        # Sort by TCE (highest first)
        matches.sort(key=lambda x: x["tce_analysis"]["tce_usd_per_day"], reverse=True)
        # Convert to serializable format
        return [self._convert_to_serializable(match) for match in matches]
    
    def find_vessel_matches(self, 
                           cargo_id: str,
                           min_tce_usd_per_day: float = 5000,
                           vessel_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Find suitable vessels for a specific cargo.
        
        Args:
            cargo_id: Cargo ID
            min_tce_usd_per_day: Minimum TCE required
            vessel_types: List of preferred vessel types
            
        Returns:
            List of vessel matches sorted by TCE
        """
        # Get cargo details
        cargo = self.data.cargos[self.data.cargos["cargo_id"] == cargo_id]
        if cargo.empty:
            raise ValueError(f"Cargo {cargo_id} not found")
        
        cargo_data = cargo.iloc[0]
        cargo_qty = cargo_data["qty_mt"]
        
        # Filter vessels by DWT compatibility (±10%)
        min_dwt = cargo_qty * 0.9
        max_dwt = cargo_qty * 1.1
        
        vessels = self.data.get_vessels(
            min_dwt=min_dwt,
            max_dwt=max_dwt
        )
        
        if vessel_types:
            vessels = [v for v in vessels if v["type"] in vessel_types]
        
        matches = []
        
        for vessel in vessels:
            try:
                # Plan voyage for this vessel
                voyage_plan = self.voyage_planner.plan_voyage(
                    vessel_imo=vessel["imo"],
                    load_port=cargo_data["load_port"],
                    disch_port=cargo_data["disch_port"],
                    speed_knots=14.0  # Default speed
                )
                
                # Get freight rate estimate
                route_key = f"{cargo_data['load_port']}-{cargo_data['disch_port']}"
                freight_rate = self.data.get_freight_rate_estimate(vessel["type"], route_key)
                if not freight_rate:
                    freight_rate = 30.0  # Default rate
                
                # Calculate TCE
                tce_analysis = self.voyage_planner.calculate_tce(
                    voyage_plan=voyage_plan,
                    freight_rate_usd_per_mt=freight_rate,
                    cargo_quantity_mt=cargo_qty
                )
                
                # Check if TCE meets minimum requirement
                if tce_analysis["tce_usd_per_day"] >= min_tce_usd_per_day:
                    match = {
                        "vessel_imo": vessel["imo"],
                        "vessel_name": vessel["name"],
                        "vessel_type": vessel["type"],
                        "dwt": vessel["dwt"],
                        "voyage_plan": voyage_plan,
                        "freight_rate_usd_per_mt": freight_rate,
                        "tce_analysis": tce_analysis,
                        "cargo_compatibility_score": self._calculate_compatibility_score(vessel, cargo_data)
                    }
                    matches.append(match)
                
            except Exception as e:
                print(f"Error processing vessel {vessel['imo']}: {e}")
                continue
        
        # Sort by TCE (highest first)
        matches.sort(key=lambda x: x["tce_analysis"]["tce_usd_per_day"], reverse=True)
        # Convert to serializable format
        return [self._convert_to_serializable(match) for match in matches]
    
    def get_optimal_matches(self, 
                           min_tce_usd_per_day: float = 5000,
                           vessel_types: Optional[List[str]] = None,
                           cargo_types: Optional[List[str]] = None,
                           max_matches: int = 20) -> List[Dict]:
        """
        Find optimal vessel-cargo combinations across the entire fleet.
        
        Returns:
            List of optimal matches sorted by TCE
        """
        all_matches = []
        
        # Get all vessels
        vessels = self.data.get_vessels()
        if vessel_types:
            vessels = [v for v in vessels if v["type"] in vessel_types]
        
        # Get all cargoes
        cargoes = self.data.get_cargos()
        if cargo_types:
            cargoes = [c for c in cargoes if c["commodity"] in cargo_types]
        
        # Find matches for each vessel-cargo combination
        for vessel in vessels:
            for cargo in cargoes:
                try:
                    # Check DWT compatibility
                    if not (cargo["qty_mt"] * 0.9 <= vessel["dwt"] <= cargo["qty_mt"] * 1.1):
                        continue
                    
                    # Plan voyage
                    voyage_plan = self.voyage_planner.plan_voyage(
                        vessel_imo=vessel["imo"],
                        load_port=cargo["load_port"],
                        disch_port=cargo["disch_port"],
                        speed_knots=14.0
                    )
                    
                    # Get freight rate
                    route_key = f"{cargo['load_port']}-{cargo['disch_port']}"
                    freight_rate = self.data.get_freight_rate_estimate(vessel["type"], route_key)
                    if not freight_rate:
                        freight_rate = 30.0
                    
                    # Calculate TCE
                    tce_analysis = self.voyage_planner.calculate_tce(
                        voyage_plan=voyage_plan,
                        freight_rate_usd_per_mt=freight_rate,
                        cargo_quantity_mt=cargo["qty_mt"]
                    )
                    
                    if tce_analysis["tce_usd_per_day"] >= min_tce_usd_per_day:
                        match = {
                            "vessel_imo": vessel["imo"],
                            "vessel_name": vessel["name"],
                            "vessel_type": vessel["type"],
                            "dwt": vessel["dwt"],
                            "cargo_id": cargo["cargo_id"],
                            "commodity": cargo["commodity"],
                            "quantity_mt": cargo["qty_mt"],
                            "load_port": cargo["load_port"],
                            "disch_port": cargo["disch_port"],
                            "laycan_open": cargo["laycan_open"].isoformat() if hasattr(cargo["laycan_open"], 'isoformat') else str(cargo["laycan_open"]),
                            "laycan_close": cargo["laycan_close"].isoformat() if hasattr(cargo["laycan_close"], 'isoformat') else str(cargo["laycan_close"]),
                            "voyage_plan": voyage_plan,
                            "freight_rate_usd_per_mt": freight_rate,
                            "tce_analysis": tce_analysis,
                            "compatibility_score": self._calculate_compatibility_score(vessel, cargo)
                        }
                        all_matches.append(match)
                
                except Exception as e:
                    continue
        
        # Sort by TCE and return top matches
        all_matches.sort(key=lambda x: x["tce_analysis"]["tce_usd_per_day"], reverse=True)
        # Convert to serializable format
        return [self._convert_to_serializable(match) for match in all_matches[:max_matches]]
    
    def analyze_market_opportunities(self, 
                                   vessel_type: str,
                                   region: Optional[str] = None) -> Dict:
        """
        Analyze market opportunities for a specific vessel type.
        
        Returns:
            Dictionary with market analysis
        """
        # Get vessels of specified type
        vessels = self.data.get_vessels(vessel_type=vessel_type)
        
        # Get cargoes
        cargoes = self.data.get_cargos()
        if region:
            # Filter by region (simplified - would need port region mapping)
            pass
        
        opportunities = []
        total_tce = 0
        valid_combinations = 0
        
        for vessel in vessels:
            for cargo in cargoes:
                try:
                    # Check compatibility
                    if not (cargo["qty_mt"] * 0.9 <= vessel["dwt"] <= cargo["qty_mt"] * 1.1):
                        continue
                    
                    # Quick TCE calculation
                    route_key = f"{cargo['load_port']}-{cargo['disch_port']}"
                    freight_rate = self.data.get_freight_rate_estimate(vessel_type, route_key) or 30.0
                    
                    # Simplified cost estimation
                    distance = self.voyage_planner.calculate_distance(cargo["load_port"], cargo["disch_port"]) or 1000
                    voyage_days = distance / (14 * 24)  # 14 knots
                    voyage_cost = voyage_days * 5000  # Simplified cost per day
                    voyage_revenue = freight_rate * cargo["qty_mt"]
                    
                    tce = (voyage_revenue - voyage_cost) / voyage_days if voyage_days > 0 else 0
                    
                    if tce > 0:
                        opportunities.append({
                            "vessel_imo": vessel["imo"],
                            "cargo_id": cargo["cargo_id"],
                            "tce_usd_per_day": tce,
                            "route": f"{cargo['load_port']}-{cargo['disch_port']}"
                        })
                        total_tce += tce
                        valid_combinations += 1
                
                except Exception:
                    continue
        
        avg_tce = total_tce / valid_combinations if valid_combinations > 0 else 0
        
        return self._convert_to_serializable({
            "vessel_type": vessel_type,
            "total_opportunities": len(opportunities),
            "average_tce_usd_per_day": avg_tce,
            "top_opportunities": sorted(opportunities, key=lambda x: x["tce_usd_per_day"], reverse=True)[:10],
            "market_summary": {
                "total_vessels": len(vessels),
                "total_cargoes": len(cargoes),
                "valid_combinations": valid_combinations
            }
        })
    
    def _estimate_ballast_distance(self, load_port: str, max_distance: float) -> float:
        """Estimate ballast distance to load port (simplified)."""
        # In reality, this would be calculated from current vessel position
        # For now, return a random distance within the limit
        import random
        return random.uniform(100, max_distance)
    
    def _calculate_compatibility_score(self, vessel: Dict, cargo: Dict) -> float:
        """Calculate compatibility score between vessel and cargo (0-100)."""
        score = 100.0
        
        # DWT compatibility (40% weight)
        dwt_ratio = min(vessel["dwt"] / cargo["qty_mt"], cargo["qty_mt"] / vessel["dwt"])
        dwt_score = min(dwt_ratio * 100, 100)
        score = score * 0.6 + dwt_score * 0.4
        
        # Vessel type vs cargo type compatibility (30% weight)
        # This would be based on historical data and market preferences
        type_compatibility = {
            ("Capesize", "Iron Ore"): 95,
            ("Capesize", "Coal"): 90,
            ("Panamax", "Grain"): 85,
            ("Panamax", "Soybeans"): 80,
            ("Handysize", "Fertilizer"): 75,
            ("Supramax", "Cement"): 70
        }
        
        type_score = type_compatibility.get((vessel["type"], cargo["commodity"]), 60)
        score = score * 0.7 + type_score * 0.3
        
        return round(score, 1)
    
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

    def get_cargo_summary(self) -> Dict:
        """Get summary statistics of available cargoes."""
        cargoes = self.data.cargos
        
        summary = {
            "total_cargoes": len(cargoes),
            "by_commodity": self._convert_to_serializable(cargoes["commodity"].value_counts()),
            "by_region": self._convert_to_serializable(cargoes["load_port"].value_counts()),
            "quantity_stats": {
                "min_mt": self._convert_to_serializable(cargoes["qty_mt"].min()),
                "max_mt": self._convert_to_serializable(cargoes["qty_mt"].max()),
                "avg_mt": self._convert_to_serializable(cargoes["qty_mt"].mean()),
                "median_mt": self._convert_to_serializable(cargoes["qty_mt"].median())
            },
            "laycan_stats": {
                "earliest": self._convert_to_serializable(cargoes["laycan_open"].min()),
                "latest": self._convert_to_serializable(cargoes["laycan_close"].max())
            }
        }
        
        return self._convert_to_serializable(summary)
    
    def get_vessel_summary(self) -> Dict:
        """Get summary statistics of available vessels."""
        vessels = self.data.vessels
        
        summary = {
            "total_vessels": len(vessels),
            "by_type": self._convert_to_serializable(vessels["type"].value_counts()),
            "dwt_stats": {
                "min_dwt": self._convert_to_serializable(vessels["dwt"].min()),
                "max_dwt": self._convert_to_serializable(vessels["dwt"].max()),
                "avg_dwt": self._convert_to_serializable(vessels["dwt"].mean()),
                "median_dwt": self._convert_to_serializable(vessels["dwt"].median())
            },
            "consumption_stats": {
                "avg_13kn": self._convert_to_serializable(vessels["cons_13_kn_mt_per_day"].mean()),
                "avg_14_5kn": self._convert_to_serializable(vessels["cons_14_5_kn_mt_per_day"].mean()),
                "avg_16kn": self._convert_to_serializable(vessels["cons_16_kn_mt_per_day"].mean())
            }
        }
        
        return self._convert_to_serializable(summary)
