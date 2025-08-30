from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .data_loader import DataLoader
from .voyage_planner import VoyagePlanner

class PDACalculator:
    """Handles Port Disbursement Account (PDA) calculations and cost analysis."""
    
    def __init__(self):
        self.data = DataLoader()
        self.voyage_planner = VoyagePlanner()
    
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
    
    def calculate_pda(self, 
                     voyage_plan: Dict,
                     bunker_port: Optional[str] = None,
                     fuel_type: str = "VLSFO",
                     include_agency_fees: bool = True,
                     include_stevedoring: bool = True) -> Dict:
        """
        Calculate comprehensive PDA for a voyage.
        
        Args:
            voyage_plan: Voyage plan from VoyagePlanner
            bunker_port: Port for bunkering (if different from load port)
            fuel_type: Type of fuel (VLSFO, HSFO, LSMGO)
            include_agency_fees: Whether to include agency fees
            include_stevedoring: Whether to include stevedoring costs
            
        Returns:
            Dictionary with detailed PDA breakdown
        """
        load_port = voyage_plan["load_port"]
        disch_port = voyage_plan["disch_port"]
        
        # Get port costs
        load_port_costs = self.data.get_port_costs(load_port)
        disch_port_costs = self.data.get_port_costs(disch_port)
        
        if not load_port_costs or not disch_port_costs:
            raise ValueError(f"Port cost data not available for {load_port} or {disch_port}")
        
        # Calculate port fees
        load_port_fees = self._calculate_port_fees(load_port_costs, include_agency_fees, include_stevedoring)
        disch_port_fees = self._calculate_port_fees(disch_port_costs, include_agency_fees, include_stevedoring)
        
        # Bunker costs
        bunker_analysis = self._calculate_bunker_costs(
            voyage_plan, 
            bunker_port or load_port, 
            fuel_type
        )
        
        # Canal costs (if applicable)
        canal_costs = voyage_plan.get("canal_cost_usd", 0)
        
        # Additional costs
        additional_costs = self._calculate_additional_costs(voyage_plan)
        
        # Total PDA
        total_pda = (
            load_port_fees["total"] +
            disch_port_fees["total"] +
            bunker_analysis["total_cost"] +
            canal_costs +
            additional_costs["total"]
        )
        
        result = {
            "voyage_summary": {
                "vessel_name": voyage_plan["vessel_name"],
                "load_port": load_port,
                "disch_port": disch_port,
                "voyage_days": voyage_plan["total_voyage_days"],
                "distance_nm": voyage_plan["distance_nm"]
            },
            "load_port_fees": load_port_fees,
            "disch_port_fees": disch_port_fees,
            "bunker_costs": bunker_analysis,
            "canal_costs": {
                "canal_toll_usd": canal_costs,
                "canal_delay_days": voyage_plan.get("canal_delay_days", 0)
            },
            "additional_costs": additional_costs,
            "total_pda_usd": total_pda,
            "cost_breakdown": {
                "port_fees_percent": round((load_port_fees["total"] + disch_port_fees["total"]) / total_pda * 100, 1),
                "bunker_percent": round(bunker_analysis["total_cost"] / total_pda * 100, 1),
                "canal_percent": round(canal_costs / total_pda * 100, 1),
                "additional_percent": round(additional_costs["total"] / total_pda * 100, 1)
            },
            "budget_analysis": self._analyze_budget(total_pda, voyage_plan),
            "cost_optimization": self._get_cost_optimization_recommendations(
                load_port_fees, disch_port_fees, bunker_analysis, canal_costs
            )
        }
        return self._convert_to_serializable(result)
    
    def compare_bunker_ports(self, 
                           voyage_plan: Dict,
                           candidate_ports: List[str],
                           fuel_type: str = "VLSFO") -> List[Dict]:
        """
        Compare bunker costs across different ports.
        
        Args:
            voyage_plan: Voyage plan
            candidate_ports: List of ports to compare
            fuel_type: Type of fuel
            
        Returns:
            List of bunker port comparisons sorted by total cost
        """
        comparisons = []
        
        for port in candidate_ports:
            try:
                bunker_analysis = self._calculate_bunker_costs(voyage_plan, port, fuel_type)
                
                comparison = {
                    "port": port,
                    "fuel_type": fuel_type,
                    "price_usd_per_mt": bunker_analysis["price_per_mt"],
                    "total_cost_usd": bunker_analysis["total_cost"],
                    "savings_vs_load_port": 0,  # Will be calculated below
                    "recommendation": bunker_analysis.get("recommendation", "")
                }
                comparisons.append(comparison)
                
            except Exception as e:
                print(f"Error analyzing bunker port {port}: {e}")
                continue
        
        # Calculate savings vs load port
        load_port_bunker = self._calculate_bunker_costs(voyage_plan, voyage_plan["load_port"], fuel_type)
        load_port_cost = load_port_bunker["total_cost"]
        
        for comparison in comparisons:
            comparison["savings_vs_load_port"] = load_port_cost - comparison["total_cost_usd"]
        
        # Sort by total cost (lowest first)
        comparisons.sort(key=lambda x: x["total_cost_usd"])
        
        return comparisons
    
    def estimate_laytime_costs(self, 
                             port: str,
                             cargo_type: str,
                             cargo_quantity_mt: float,
                             demurrage_rate_usd_per_day: float = 25000) -> Dict:
        """
        Estimate laytime costs including demurrage/despatch.
        
        Args:
            port: Port code
            cargo_type: Type of cargo
            cargo_quantity_mt: Cargo quantity in MT
            demurrage_rate_usd_per_day: Demurrage rate per day
            
        Returns:
            Dictionary with laytime cost analysis
        """
        # Get laytime rate
        laytime_rate = self.data.get_laytime_rate(port, cargo_type)
        if not laytime_rate:
            laytime_rate = 1000  # Default rate
        
        # Calculate laytime
        laytime_days = cargo_quantity_mt / laytime_rate
        
        # Get port costs for laytime calculation
        port_costs = self.data.get_port_costs(port)
        if not port_costs:
            return {"error": f"No port cost data available for {port}"}
        
        # Calculate laytime costs
        laytime_costs = {
            "estimated_laytime_days": laytime_days,
            "estimated_laytime_hours": laytime_days * 24,
            "laytime_rate_mt_per_day": laytime_rate,
            "demurrage_rate_usd_per_day": demurrage_rate_usd_per_day,
            "port_fees_during_laytime": {
                "dues_usd": port_costs.get("dues_usd", 0) * (laytime_days / 30),  # Pro-rated
                "agency_usd": port_costs.get("agency_usd", 0),
                "other_usd": port_costs.get("other_usd", 0)
            }
        }
        
        # Calculate total laytime costs
        total_laytime_costs = sum(laytime_costs["port_fees_during_laytime"].values())
        
        # Demurrage/despatch calculation (simplified)
        # Assume 2 days allowed laytime
        allowed_laytime = 2.0
        if laytime_days > allowed_laytime:
            demurrage_days = laytime_days - allowed_laytime
            demurrage_cost = demurrage_days * demurrage_rate_usd_per_day
            laytime_costs["demurrage_cost_usd"] = demurrage_cost
            laytime_costs["despatch_cost_usd"] = 0
        else:
            despatch_days = allowed_laytime - laytime_days
            despatch_rate = demurrage_rate_usd_per_day * 0.5  # Half rate for despatch
            despatch_cost = despatch_days * despatch_rate
            laytime_costs["demurrage_cost_usd"] = 0
            laytime_costs["despatch_cost_usd"] = despatch_cost
        
        laytime_costs["total_laytime_cost_usd"] = (
            total_laytime_costs + 
            laytime_costs.get("demurrage_cost_usd", 0) - 
            laytime_costs.get("despatch_cost_usd", 0)
        )
        
        return laytime_costs
    
    def get_cost_benchmarks(self, vessel_type: str, route: str) -> Dict:
        """
        Get cost benchmarks for a vessel type and route.
        
        Args:
            vessel_type: Vessel type
            route: Route identifier
            
        Returns:
            Dictionary with cost benchmarks
        """
        # Simplified benchmarks based on vessel type and route
        benchmarks = {
            "Capesize": {
                "BRSSZ-CNSHA": {"avg_pda": 850000, "avg_port_fees": 120000, "avg_bunker": 650000},
                "AUPHE-CNSHA": {"avg_pda": 750000, "avg_port_fees": 110000, "avg_bunker": 580000},
                "default": {"avg_pda": 800000, "avg_port_fees": 115000, "avg_bunker": 620000}
            },
            "Panamax": {
                "BRSSZ-CNSHA": {"avg_pda": 450000, "avg_port_fees": 80000, "avg_bunker": 320000},
                "USGDX-CNSHA": {"avg_pda": 500000, "avg_port_fees": 85000, "avg_bunker": 350000},
                "default": {"avg_pda": 475000, "avg_port_fees": 82500, "avg_bunker": 335000}
            },
            "Handysize": {
                "default": {"avg_pda": 250000, "avg_port_fees": 50000, "avg_bunker": 180000}
            }
        }
        
        vessel_benchmarks = benchmarks.get(vessel_type, benchmarks["Panamax"])
        route_benchmarks = vessel_benchmarks.get(route, vessel_benchmarks["default"])
        
        return {
            "vessel_type": vessel_type,
            "route": route,
            "benchmarks": route_benchmarks,
            "cost_per_day": route_benchmarks["avg_pda"] / 30,  # Assume 30-day voyage
            "cost_per_nm": route_benchmarks["avg_pda"] / 8000  # Assume 8000 NM voyage
        }
    
    def _calculate_port_fees(self, port_costs: Dict, include_agency: bool, include_stevedoring: bool) -> Dict:
        """Calculate detailed port fees breakdown."""
        fees = {
            "dues_usd": port_costs.get("dues_usd", 0),
            "pilotage_usd": port_costs.get("pilotage_usd", 0),
            "towage_usd": port_costs.get("towage_usd", 0),
            "mooring_usd": port_costs.get("mooring_usd", 0),
            "agency_usd": port_costs.get("agency_usd", 0) if include_agency else 0,
            "stevedoring_usd": port_costs.get("stevedoring_usd", 0) if include_stevedoring else 0,
            "other_usd": port_costs.get("other_usd", 0)
        }
        
        fees["total"] = sum(fees.values())
        return fees
    
    def _calculate_bunker_costs(self, voyage_plan: Dict, bunker_port: str, fuel_type: str) -> Dict:
        """Calculate bunker costs for the voyage."""
        # Get bunker price
        bunker_price = self.data.get_bunker_price(bunker_port, fuel_type)
        if not bunker_price:
            bunker_price = 600  # Default price
        
        # Calculate fuel consumption
        fuel_consumption_mt = voyage_plan["fuel_consumption_mt"]
        
        # Calculate total cost
        total_cost = fuel_consumption_mt * bunker_price
        
        # Get price trend
        price_analysis = self._analyze_bunker_price_trend(bunker_port, fuel_type)
        
        return {
            "bunker_port": bunker_port,
            "fuel_type": fuel_type,
            "price_per_mt": bunker_price,
            "consumption_mt": fuel_consumption_mt,
            "total_cost": total_cost,
            "price_trend": price_analysis.get("trend", "stable"),
            "recommendation": price_analysis.get("recommendation", "")
        }
    
    def _calculate_additional_costs(self, voyage_plan: Dict) -> Dict:
        """Calculate additional voyage costs."""
        additional_costs = {
            "insurance_usd": voyage_plan.get("distance_nm", 0) * 0.5,  # $0.5 per NM
            "communication_usd": voyage_plan.get("total_voyage_days", 0) * 50,  # $50 per day
            "provisions_usd": voyage_plan.get("total_voyage_days", 0) * 100,  # $100 per day
            "lubricants_usd": voyage_plan.get("fuel_consumption_mt", 0) * 0.05,  # 5% of fuel cost
            "miscellaneous_usd": 5000  # Fixed amount
        }
        
        additional_costs["total"] = sum(additional_costs.values())
        return additional_costs
    
    def _analyze_bunker_price_trend(self, port: str, fuel_type: str) -> Dict:
        """Analyze bunker price trend for recommendations."""
        # Get recent price history
        recent_prices = self.data.bunker_prices[
            (self.data.bunker_prices["port"] == port) & 
            (self.data.bunker_prices["fuel"] == fuel_type)
        ].sort_values("date").tail(7)
        
        if len(recent_prices) < 2:
            return {"trend": "stable", "recommendation": "Insufficient data for trend analysis"}
        
        current_price = recent_prices.iloc[-1]["usd_per_mt"]
        week_ago_price = recent_prices.iloc[0]["usd_per_mt"]
        
        price_change = ((current_price - week_ago_price) / week_ago_price * 100) if week_ago_price > 0 else 0
        
        if price_change > 5:
            trend = "increasing"
            recommendation = "Consider forward buying or alternative ports"
        elif price_change < -5:
            trend = "decreasing"
            recommendation = "Good time to bunker, prices trending down"
        else:
            trend = "stable"
            recommendation = "Prices stable, normal procurement recommended"
        
        return {
            "trend": trend,
            "price_change_percent": round(price_change, 2),
            "recommendation": recommendation
        }
    
    def _analyze_budget(self, total_pda: float, voyage_plan: Dict) -> Dict:
        """Analyze PDA against budget and provide insights."""
        # Get benchmarks for comparison
        benchmarks = self.get_cost_benchmarks(
            voyage_plan["vessel_type"], 
            f"{voyage_plan['load_port']}-{voyage_plan['disch_port']}"
        )
        
        avg_pda = benchmarks["benchmarks"]["avg_pda"]
        budget_variance = ((total_pda - avg_pda) / avg_pda * 100) if avg_pda > 0 else 0
        
        if budget_variance < -10:
            status = "Under Budget"
            assessment = "Excellent cost control"
        elif budget_variance < 0:
            status = "Slightly Under Budget"
            assessment = "Good cost management"
        elif budget_variance < 10:
            status = "Within Budget"
            assessment = "Costs in line with expectations"
        else:
            status = "Over Budget"
            assessment = "Review costs and identify optimization opportunities"
        
        return {
            "total_pda_usd": total_pda,
            "benchmark_pda_usd": avg_pda,
            "budget_variance_percent": round(budget_variance, 1),
            "status": status,
            "assessment": assessment,
            "cost_per_day": total_pda / voyage_plan["total_voyage_days"] if voyage_plan["total_voyage_days"] > 0 else 0
        }
    
    def _get_cost_optimization_recommendations(self, 
                                             load_fees: Dict, 
                                             disch_fees: Dict, 
                                             bunker_costs: Dict, 
                                             canal_costs: float) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        # Port fees optimization
        total_port_fees = load_fees["total"] + disch_fees["total"]
        if total_port_fees > 150000:  # High port fees
            recommendations.append("Review port fee negotiations and seek competitive quotes")
            recommendations.append("Consider alternative ports with lower fees")
        
        # Bunker optimization
        if bunker_costs["price_per_mt"] > 650:  # High bunker price
            recommendations.append("Explore alternative bunker ports with better prices")
            recommendations.append("Consider fuel hedging strategies")
        
        # Canal optimization
        if canal_costs > 400000:  # High canal costs
            recommendations.append("Evaluate alternative routes to avoid canal fees")
            recommendations.append("Consider vessel size optimization for canal transit")
        
        # General recommendations
        recommendations.append("Implement fuel consumption monitoring and optimization")
        recommendations.append("Review agency fee negotiations")
        recommendations.append("Consider bulk purchasing for provisions and supplies")
        
        return recommendations
