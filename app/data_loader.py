import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

class DataLoader:
    """Loads and manages all maritime data from CSV files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all CSV files into memory."""
        try:
            # Load main data files
            self.vessels = pd.read_csv(self.data_dir / "vessels.csv")
            self.cargos = pd.read_csv(self.data_dir / "cargos.csv")
            self.ports = pd.read_csv(self.data_dir / "ports.csv")
            self.routes = pd.read_csv(self.data_dir / "routes.csv")
            self.bunker_prices = pd.read_csv(self.data_dir / "bunker_prices.csv")
            self.port_costs = pd.read_csv(self.data_dir / "port_costs.csv")
            self.indices = pd.read_csv(self.data_dir / "indices.csv")
            self.laytime_rates = pd.read_csv(self.data_dir / "laytime_rates.csv")
            self.canal_fees = pd.read_csv(self.data_dir / "canal_fees.csv")
            self.ais_tracks = pd.read_csv(self.data_dir / "ais_tracks.csv")
            
            # Convert date columns
            self.bunker_prices['date'] = pd.to_datetime(self.bunker_prices['date'])
            self.indices['date'] = pd.to_datetime(self.indices['date'])
            self.cargos['laycan_open'] = pd.to_datetime(self.cargos['laycan_open'])
            self.cargos['laycan_close'] = pd.to_datetime(self.cargos['laycan_close'])
            
            # Convert IMO numbers to strings for consistent comparison
            self.vessels['imo'] = self.vessels['imo'].astype(str)
            self.ais_tracks['imo'] = self.ais_tracks['imo'].astype(str)
            
            print(f"✅ Loaded {len(self.vessels)} vessels, {len(self.cargos)} cargoes, {len(self.ports)} ports")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def get_vessels(self, 
                   vessel_type: Optional[str] = None, 
                   min_dwt: Optional[float] = None, 
                   max_dwt: Optional[float] = None,
                   current_port: Optional[str] = None) -> List[Dict]:
        """Filter vessels by type, DWT, and current port."""
        df = self.vessels.copy()
        
        if vessel_type:
            df = df[df["type"] == vessel_type]
        if min_dwt:
            df = df[df["dwt"] >= min_dwt]
        if max_dwt:
            df = df[df["dwt"] <= max_dwt]
        if current_port:
            # For now, we'll use a simple approach - in real scenario, this would come from AIS
            pass
            
        return df.to_dict("records")
    
    def get_cargos(self, 
                   cargo_type: Optional[str] = None, 
                   load_port: Optional[str] = None,
                   min_quantity: Optional[float] = None,
                   max_quantity: Optional[float] = None,
                   laycan_start: Optional[datetime] = None,
                   laycan_end: Optional[datetime] = None) -> List[Dict]:
        """Filter cargoes by type, port, quantity, and laycan dates."""
        df = self.cargos.copy()
        
        if cargo_type:
            df = df[df["commodity"] == cargo_type]
        if load_port:
            df = df[df["load_port"] == load_port]
        if min_quantity:
            df = df[df["qty_mt"] >= min_quantity]
        if max_quantity:
            df = df[df["qty_mt"] <= max_quantity]
        if laycan_start:
            df = df[df["laycan_open"] >= laycan_start]
        if laycan_end:
            df = df[df["laycan_close"] <= laycan_end]
            
        return df.to_dict("records")
    
    def get_port_details(self, port_code: str) -> Optional[Dict]:
        """Get complete port details including costs and coordinates."""
        port_info = self.ports[self.ports["code"] == port_code]
        if port_info.empty:
            return None
            
        port_data = port_info.iloc[0].to_dict()
        
        # Add port costs
        port_costs = self.port_costs[self.port_costs["port"] == port_code]
        if not port_costs.empty:
            cost_data = port_costs.iloc[0].to_dict()
            port_data.update(cost_data)
        
        # Add latest bunker prices
        latest_bunker = self.bunker_prices[
            self.bunker_prices["port"] == port_code
        ].sort_values("date").tail(1)
        
        if not latest_bunker.empty:
            port_data["latest_vlsfo_price"] = latest_bunker.iloc[0]["usd_per_mt"]
        
        return port_data
    
    def get_route(self, load_port: str, disch_port: str, variant: str = "DIRECT") -> Optional[Dict]:
        """Get route information between two ports."""
        # First try the direct route
        route = self.routes[
            (self.routes["load"] == load_port) & 
            (self.routes["disch"] == disch_port) & 
            (self.routes["variant"] == variant)
        ]
        
        if not route.empty:
            return route.iloc[0].to_dict()
        
        # If not found, try to find a reverse route and create a reverse version
        reverse_route = self.routes[
            (self.routes["load"] == disch_port) & 
            (self.routes["disch"] == load_port) & 
            (self.routes["variant"] == variant)
        ]
        
        if not reverse_route.empty:
            # Create a reverse route with the same distance and costs
            reverse_data = reverse_route.iloc[0].to_dict()
            return {
                "load": load_port,
                "disch": disch_port,
                "variant": variant,
                "distance_nm": reverse_data["distance_nm"],
                "canal_delay_days": reverse_data["canal_delay_days"],
                "canal_toll_usd": reverse_data["canal_toll_usd"],
                "piracy_risk": reverse_data["piracy_risk"],
                "eca_nm": reverse_data["eca_nm"]
            }
        
        return None
    
    def get_all_routes(self, load_port: str, disch_port: str) -> List[Dict]:
        """Get all route variants between two ports."""
        routes = self.routes[
            (self.routes["load"] == load_port) & 
            (self.routes["disch"] == disch_port)
        ]
        
        # Also check for reverse routes
        reverse_routes = self.routes[
            (self.routes["load"] == disch_port) & 
            (self.routes["disch"] == load_port)
        ]
        
        all_routes = []
        
        # Add direct routes
        for _, route in routes.iterrows():
            all_routes.append(route.to_dict())
        
        # Add reverse routes (with corrected direction)
        for _, route in reverse_routes.iterrows():
            reverse_data = route.to_dict()
            all_routes.append({
                "load": load_port,
                "disch": disch_port,
                "variant": reverse_data["variant"],
                "distance_nm": reverse_data["distance_nm"],
                "canal_delay_days": reverse_data["canal_delay_days"],
                "canal_toll_usd": reverse_data["canal_toll_usd"],
                "piracy_risk": reverse_data["piracy_risk"],
                "eca_nm": reverse_data["eca_nm"]
            })
        
        return all_routes
    
    def get_bunker_price(self, port: str, fuel_type: str = "VLSFO", date: Optional[datetime] = None) -> Optional[float]:
        """Get bunker price for a port and fuel type."""
        prices = self.bunker_prices[
            (self.bunker_prices["port"] == port) & 
            (self.bunker_prices["fuel"] == fuel_type)
        ]
        
        if prices.empty:
            return None
            
        if date:
            # Get price closest to the specified date
            prices = prices.sort_values("date")
            closest_price = prices.iloc[(prices['date'] - date).abs().argsort()[:1]]
            return closest_price.iloc[0]["usd_per_mt"] if not closest_price.empty else None
        else:
            # Get latest price
            return prices.sort_values("date").iloc[-1]["usd_per_mt"]
    
    def get_market_indices(self, date: Optional[datetime] = None) -> Optional[Dict]:
        """Get market indices (BDI, VLSFO) for a specific date or latest."""
        if date:
            indices = self.indices[self.indices["date"] <= date].sort_values("date")
        else:
            indices = self.indices.sort_values("date")
            
        if indices.empty:
            return None
            
        return indices.iloc[-1].to_dict()
    
    def get_vessel_consumption(self, vessel_imo: str, speed_knots: float) -> Optional[Dict]:
        """Get fuel consumption for a vessel at a specific speed."""
        # Convert vessel_imo to string for comparison
        vessel_imo_str = str(vessel_imo)
        vessel = self.vessels[self.vessels["imo"] == vessel_imo_str]
        if vessel.empty:
            return None
            
        vessel_data = vessel.iloc[0]
        
        # Interpolate consumption based on speed
        if speed_knots <= 13:
            consumption = vessel_data["cons_13_kn_mt_per_day"]
        elif speed_knots <= 14.5:
            consumption = vessel_data["cons_14_5_kn_mt_per_day"]
        elif speed_knots <= 16:
            consumption = vessel_data["cons_16_kn_mt_per_day"]
        else:
            # Extrapolate for higher speeds
            consumption = vessel_data["cons_16_kn_mt_per_day"] * (speed_knots / 16) ** 2
        
        return {
            "main_consumption_mt_per_day": consumption,
            "aux_consumption_mt_per_day": vessel_data["aux_port_mt_per_day"],
            "eca_penalty_mt_per_day": vessel_data["eca_penalty_mt_per_day"]
        }
    
    def get_port_costs(self, port: str) -> Optional[Dict]:
        """Get all port costs for a specific port."""
        costs = self.port_costs[self.port_costs["port"] == port]
        if costs.empty:
            return None
            
        return costs.iloc[0].to_dict()
    
    def get_canal_fee(self, canal: str, vessel_dwt: float) -> Optional[float]:
        """Get canal transit fee based on vessel DWT."""
        canal_data = self.canal_fees[self.canal_fees["canal"] == canal]
        if canal_data.empty:
            return None
            
        # Simple linear interpolation based on DWT
        # In reality, canal fees have complex tiered structures
        base_fee = canal_data.iloc[0]["base_fee_usd"]
        dwt_factor = vessel_dwt / 100000  # Normalize to 100k DWT
        return base_fee * dwt_factor
    
    def get_laytime_rate(self, port: str, cargo_type: str) -> Optional[float]:
        """Get laytime rate for a port and cargo type."""
        rates = self.laytime_rates[
            (self.laytime_rates["port"] == port) & 
            (self.laytime_rates["cargo_type"] == cargo_type)
        ]
        
        if rates.empty:
            return None
            
        return rates.iloc[0]["rate_mt_per_day"]
    
    def get_ais_position(self, vessel_imo: str) -> Optional[Dict]:
        """Get latest AIS position for a vessel."""
        vessel_imo_str = str(vessel_imo)
        ais_data = self.ais_tracks[self.ais_tracks["imo"] == vessel_imo_str]
        if ais_data.empty:
            return None
            
        # Get latest position
        latest = ais_data.sort_values("timestamp").iloc[-1]
        return latest.to_dict()
    
    def get_nearby_vessels(self, port: str, radius_nm: float = 100) -> List[Dict]:
        """Get vessels within a certain radius of a port."""
        # This is a simplified implementation
        # In reality, you'd calculate actual distances from AIS positions
        port_data = self.get_port_details(port)
        if not port_data:
            return []
        
        # For demo purposes, return random vessels
        # In production, you'd filter by actual distance calculations
        return self.vessels.sample(min(5, len(self.vessels))).to_dict("records")
    
    def get_freight_rate_estimate(self, vessel_type: str, route: str) -> Optional[float]:
        """Get freight rate estimate based on vessel type and route."""
        # This would typically come from freight indices or market data
        # For now, we'll use a simple estimation based on BDI
        latest_indices = self.get_market_indices()
        if not latest_indices:
            return None
            
        bdi = latest_indices["BDI"]
        
        # Simple rate estimation (in USD/MT)
        base_rates = {
            "Capesize": 25.0,
            "Panamax": 35.0,
            "Supramax": 30.0,
            "Handysize": 28.0,
            "Kamsarmax": 32.0
        }
        
        base_rate = base_rates.get(vessel_type, 30.0)
        bdi_factor = bdi / 2000  # Normalize to BDI 2000
        
        return base_rate * bdi_factor
