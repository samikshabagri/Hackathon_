from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .data_loader import DataLoader

class MarketInsights:
    """Provides market analysis, freight rate trends, and benchmarking insights."""
    
    def __init__(self):
        self.data = DataLoader()
    
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
    
    def get_freight_rate_trends(self, 
                               vessel_type: str, 
                               route: str,
                               days_back: int = 30) -> Dict:
        """
        Analyze freight rate trends for a specific vessel type and route.
        
        Args:
            vessel_type: Vessel type (Capesize, Panamax, etc.)
            route: Route identifier (e.g., "BRSSZ-CNSHA")
            days_back: Number of days to look back for trend analysis
            
        Returns:
            Dictionary with trend analysis
        """
        # Get historical indices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        historical_indices = self.data.indices[
            (self.data.indices["date"] >= start_date) & 
            (self.data.indices["date"] <= end_date)
        ].sort_values("date")
        
        if historical_indices.empty:
            return {"error": "No historical data available"}
        
        # Calculate trend metrics
        current_bdi = historical_indices.iloc[-1]["BDI"]
        current_vlsfo = historical_indices.iloc[-1]["VLSFO_index_usd_mt"]
        
        # Calculate changes
        if len(historical_indices) > 1:
            week_ago_bdi = historical_indices.iloc[-8]["BDI"] if len(historical_indices) >= 8 else historical_indices.iloc[0]["BDI"]
            month_ago_bdi = historical_indices.iloc[0]["BDI"]
            
            bdi_change_week = ((current_bdi - week_ago_bdi) / week_ago_bdi * 100) if week_ago_bdi > 0 else 0
            bdi_change_month = ((current_bdi - month_ago_bdi) / month_ago_bdi * 100) if month_ago_bdi > 0 else 0
        else:
            bdi_change_week = bdi_change_month = 0
        
        # Estimate freight rate based on BDI
        base_rates = {
            "Capesize": 25.0,
            "Panamax": 35.0,
            "Supramax": 30.0,
            "Handysize": 28.0,
            "Kamsarmax": 32.0
        }
        
        base_rate = base_rates.get(vessel_type, 30.0)
        bdi_factor = current_bdi / 2000  # Normalize to BDI 2000
        estimated_rate = base_rate * bdi_factor
        
        # Route-specific adjustments
        route_adjustments = {
            "BRSSZ-CNSHA": 1.1,  # Brazil-China premium
            "AUPHE-CNSHA": 1.05,  # Australia-China
            "USGDX-CNSHA": 1.15,  # US-China premium
            "NLRTM-CNSHA": 1.2,   # Europe-China premium
        }
        
        route_multiplier = route_adjustments.get(route, 1.0)
        adjusted_rate = estimated_rate * route_multiplier
        
        # Trend analysis
        if len(historical_indices) > 7:
            recent_trend = np.polyfit(range(len(historical_indices[-7:])), 
                                    historical_indices[-7:]["BDI"], 1)[0]
            trend_direction = "up" if recent_trend > 0 else "down"
            trend_strength = abs(recent_trend)
        else:
            trend_direction = "stable"
            trend_strength = 0
        
        result = {
            "vessel_type": vessel_type,
            "route": route,
            "current_bdi": current_bdi,
            "current_vlsfo_usd_per_mt": current_vlsfo,
            "bdi_change_week_percent": round(bdi_change_week, 2),
            "bdi_change_month_percent": round(bdi_change_month, 2),
            "estimated_freight_rate_usd_per_mt": round(adjusted_rate, 2),
            "trend_direction": trend_direction,
            "trend_strength": round(trend_strength, 2),
            "recommendation": self._get_market_recommendation(bdi_change_week, trend_direction),
            "historical_data": historical_indices.to_dict("records")
        }
        return self._convert_to_serializable(result)
    
    def benchmark_voyage_performance(self, 
                                   voyage_plan: Dict,
                                   vessel_type: str,
                                   route: str) -> Dict:
        """
        Benchmark a voyage against market averages.
        
        Args:
            voyage_plan: Voyage plan from VoyagePlanner
            vessel_type: Vessel type
            route: Route identifier
            
        Returns:
            Dictionary with benchmarking analysis
        """
        # Get market rate estimate
        market_rate = self.data.get_freight_rate_estimate(vessel_type, route)
        if not market_rate:
            market_rate = 30.0  # Default
        
        # Calculate voyage metrics
        voyage_cost = voyage_plan["total_voyage_cost_usd"]
        voyage_days = voyage_plan["total_voyage_days"]
        
        # Assume standard cargo quantity for benchmarking
        standard_cargo_qty = {
            "Capesize": 180000,
            "Panamax": 75000,
            "Supramax": 60000,
            "Handysize": 35000,
            "Kamsarmax": 85000
        }.get(vessel_type, 75000)
        
        # Calculate TCE
        voyage_revenue = market_rate * standard_cargo_qty
        tce = (voyage_revenue - voyage_cost) / voyage_days if voyage_days > 0 else 0
        
        # Get market averages (simplified)
        market_averages = {
            "Capesize": {"avg_tce": 15000, "avg_cost_per_day": 12000},
            "Panamax": {"avg_tce": 12000, "avg_cost_per_day": 8000},
            "Supramax": {"avg_tce": 10000, "avg_cost_per_day": 7000},
            "Handysize": {"avg_tce": 8000, "avg_cost_per_day": 5000},
            "Kamsarmax": {"avg_tce": 11000, "avg_cost_per_day": 7500}
        }
        
        market_avg = market_averages.get(vessel_type, {"avg_tce": 10000, "avg_cost_per_day": 7000})
        
        # Calculate performance metrics
        cost_per_day = voyage_cost / voyage_days if voyage_days > 0 else 0
        tce_performance = (tce / market_avg["avg_tce"] * 100) if market_avg["avg_tce"] > 0 else 0
        cost_performance = (cost_per_day / market_avg["avg_cost_per_day"] * 100) if market_avg["avg_cost_per_day"] > 0 else 0
        
        result = {
            "vessel_type": vessel_type,
            "route": route,
            "voyage_tce_usd_per_day": round(tce, 2),
            "market_avg_tce_usd_per_day": market_avg["avg_tce"],
            "tce_performance_percent": round(tce_performance, 1),
            "voyage_cost_per_day": round(cost_per_day, 2),
            "market_avg_cost_per_day": market_avg["avg_cost_per_day"],
            "cost_performance_percent": round(cost_performance, 1),
            "performance_rating": self._get_performance_rating(tce_performance),
            "recommendations": self._get_optimization_recommendations(tce, cost_per_day, market_avg)
        }
        return self._convert_to_serializable(result)
    
    def get_market_summary(self) -> Dict:
        """Get overall market summary and trends."""
        # Get latest indices
        latest_indices = self.data.get_market_indices()
        if not latest_indices:
            return {"error": "No market data available"}
        
        # Calculate market sentiment
        bdi = latest_indices["BDI"]
        vlsfo = latest_indices["VLSFO_index_usd_mt"]
        
        # Market sentiment based on BDI levels
        if bdi > 2500:
            sentiment = "bullish"
        elif bdi > 2000:
            sentiment = "moderate"
        elif bdi > 1500:
            sentiment = "neutral"
        else:
            sentiment = "bearish"
        
        # Get historical trends
        recent_data = self.data.indices.tail(30)
        if len(recent_data) > 1:
            bdi_trend = (recent_data.iloc[-1]["BDI"] - recent_data.iloc[0]["BDI"]) / recent_data.iloc[0]["BDI"] * 100
            vlsfo_trend = (recent_data.iloc[-1]["VLSFO_index_usd_mt"] - recent_data.iloc[0]["VLSFO_index_usd_mt"]) / recent_data.iloc[0]["VLSFO_index_usd_mt"] * 100
        else:
            bdi_trend = vlsfo_trend = 0
        
        # Vessel type analysis
        vessel_analysis = {}
        for vessel_type in ["Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"]:
            base_rate = self.data.get_freight_rate_estimate(vessel_type, "BRSSZ-CNSHA") or 30.0
            vessel_analysis[vessel_type] = {
                "estimated_rate_usd_per_mt": round(base_rate, 2),
                "market_outlook": self._get_vessel_outlook(vessel_type, bdi)
            }
        
        result = {
            "current_bdi": bdi,
            "current_vlsfo_usd_per_mt": vlsfo,
            "market_sentiment": sentiment,
            "bdi_trend_percent": round(bdi_trend, 2),
            "vlsfo_trend_percent": round(vlsfo_trend, 2),
            "vessel_analysis": vessel_analysis,
            "market_recommendations": self._get_market_recommendation(bdi_trend, "up" if bdi_trend > 0 else "down"),
            "last_updated": latest_indices["date"].isoformat() if hasattr(latest_indices["date"], 'isoformat') else str(latest_indices["date"])
        }
        return self._convert_to_serializable(result)
    
    def get_bunker_price_analysis(self, port: str, days_back: int = 30) -> Dict:
        """Analyze bunker price trends for a specific port."""
        # Get bunker price history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        port_prices = self.data.bunker_prices[
            (self.data.bunker_prices["port"] == port) & 
            (self.data.bunker_prices["fuel"] == "VLSFO") &
            (self.data.bunker_prices["date"] >= start_date) &
            (self.data.bunker_prices["date"] <= end_date)
        ].sort_values("date")
        
        if port_prices.empty:
            return {"error": f"No bunker price data available for {port}"}
        
        current_price = port_prices.iloc[-1]["usd_per_mt"]
        avg_price = port_prices["usd_per_mt"].mean()
        min_price = port_prices["usd_per_mt"].min()
        max_price = port_prices["usd_per_mt"].max()
        
        # Calculate trend
        if len(port_prices) > 1:
            price_trend = (current_price - port_prices.iloc[0]["usd_per_mt"]) / port_prices.iloc[0]["usd_per_mt"] * 100
        else:
            price_trend = 0
        
        # Compare with global average
        global_avg = self.data.bunker_prices[
            (self.data.bunker_prices["fuel"] == "VLSFO") &
            (self.data.bunker_prices["date"] >= start_date)
        ]["usd_per_mt"].mean()
        
        price_vs_global = ((current_price - global_avg) / global_avg * 100) if global_avg > 0 else 0
        
        result = {
            "port": port,
            "current_price_usd_per_mt": current_price,
            "average_price_usd_per_mt": round(avg_price, 2),
            "min_price_usd_per_mt": min_price,
            "max_price_usd_per_mt": max_price,
            "price_trend_percent": round(price_trend, 2),
            "vs_global_average_percent": round(price_vs_global, 2),
            "recommendation": self._get_bunker_recommendation(current_price, avg_price, price_trend),
            "price_history": port_prices.to_dict("records")
        }
        return self._convert_to_serializable(result)
    
    def get_route_analysis(self, load_port: str, disch_port: str) -> Dict:
        """Analyze a specific route for market opportunities."""
        # Get all route variants
        routes = self.data.get_all_routes(load_port, disch_port)
        
        if not routes:
            return {"error": f"No routes found between {load_port} and {disch_port}"}
        
        route_analysis = []
        
        for route in routes:
            # Get freight rate estimate for different vessel types
            vessel_rates = {}
            for vessel_type in ["Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"]:
                rate = self.data.get_freight_rate_estimate(vessel_type, f"{load_port}-{disch_port}")
                if rate:
                    vessel_rates[vessel_type] = round(rate, 2)
            
            route_analysis.append({
                "variant": route["variant"],
                "distance_nm": route["distance_nm"],
                "canal_toll_usd": route["canal_toll_usd"],
                "canal_delay_days": route["canal_delay_days"],
                "piracy_risk": route["piracy_risk"],
                "eca_nm": route["eca_nm"],
                "vessel_rates": vessel_rates
            })
        
        # Find best route by distance
        best_route = min(routes, key=lambda x: x["distance_nm"])
        
        result = {
            "load_port": load_port,
            "disch_port": disch_port,
            "total_routes": len(routes),
            "best_route": best_route["variant"],
            "shortest_distance_nm": best_route["distance_nm"],
            "route_analysis": route_analysis,
            "market_opportunity": self._assess_route_opportunity(load_port, disch_port)
        }
        return self._convert_to_serializable(result)
    
    def _get_market_recommendation(self, bdi_change: float, trend_direction: str) -> str:
        """Get market recommendation based on BDI changes and trend."""
        if bdi_change > 10 and trend_direction == "up":
            return "Strong buy - Market showing strong upward momentum"
        elif bdi_change > 5 and trend_direction == "up":
            return "Buy - Market trending upward"
        elif bdi_change < -10 and trend_direction == "down":
            return "Sell - Market showing strong downward pressure"
        elif bdi_change < -5 and trend_direction == "down":
            return "Hold - Market trending downward"
        else:
            return "Hold - Market relatively stable"
    
    def _get_performance_rating(self, tce_performance: float) -> str:
        """Get performance rating based on TCE performance."""
        if tce_performance >= 120:
            return "Excellent"
        elif tce_performance >= 100:
            return "Good"
        elif tce_performance >= 80:
            return "Average"
        else:
            return "Below Average"
    
    def _get_optimization_recommendations(self, tce: float, cost_per_day: float, market_avg: Dict) -> List[str]:
        """Get optimization recommendations based on performance."""
        recommendations = []
        
        if tce < market_avg["avg_tce"]:
            recommendations.append("Consider speed optimization to reduce fuel costs")
            recommendations.append("Review port costs and seek competitive quotes")
            recommendations.append("Evaluate alternative routes to reduce distance")
        
        if cost_per_day > market_avg["avg_cost_per_day"]:
            recommendations.append("Analyze fuel consumption patterns")
            recommendations.append("Review bunker procurement strategy")
            recommendations.append("Consider vessel maintenance optimization")
        
        if not recommendations:
            recommendations.append("Performance is in line with market averages")
        
        return recommendations
    
    def _get_vessel_outlook(self, vessel_type: str, bdi: float) -> str:
        """Get market outlook for a specific vessel type."""
        if vessel_type == "Capesize" and bdi > 2000:
            return "Strong - High BDI benefits Capesize vessels"
        elif vessel_type == "Panamax" and bdi > 1800:
            return "Good - Stable market conditions"
        elif vessel_type == "Handysize" and bdi > 1600:
            return "Moderate - Smaller vessels less sensitive to BDI"
        else:
            return "Challenging - Low BDI environment"
    
    def _get_bunker_recommendation(self, current_price: float, avg_price: float, trend: float) -> str:
        """Get bunker procurement recommendation."""
        if current_price < avg_price * 0.95 and trend < -5:
            return "Consider forward buying - Prices below average and trending down"
        elif current_price > avg_price * 1.05 and trend > 5:
            return "Delay bunkering if possible - Prices above average and trending up"
        else:
            return "Normal procurement - Prices within normal range"
    
    def _assess_route_opportunity(self, load_port: str, disch_port: str) -> Dict:
        """Assess market opportunity for a specific route."""
        # Simplified opportunity assessment
        # In reality, this would use more sophisticated market analysis
        
        # Get cargo volume on this route
        route_cargoes = self.data.cargos[
            (self.data.cargos["load_port"] == load_port) & 
            (self.data.cargos["disch_port"] == disch_port)
        ]
        
        total_cargo_volume = route_cargoes["qty_mt"].sum() if not route_cargoes.empty else 0
        
        # Assess opportunity level
        if total_cargo_volume > 1000000:  # 1M MT
            opportunity_level = "High"
        elif total_cargo_volume > 500000:  # 500k MT
            opportunity_level = "Medium"
        else:
            opportunity_level = "Low"
        
        return {
            "opportunity_level": opportunity_level,
            "total_cargo_volume_mt": total_cargo_volume,
            "cargo_count": len(route_cargoes),
            "primary_commodities": route_cargoes["commodity"].value_counts().head(3).to_dict() if not route_cargoes.empty else {}
        }
