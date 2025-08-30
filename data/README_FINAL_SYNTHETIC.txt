
Final High-Quality Synthetic Maritime Dataset Design  by Samiksha bagri 
=============================================
Generated: 2025-08-30T09:39:53.387922

Contents:
- vessels.csv — 150 vessels, 6 types (Handysize → Capesize + Ultramax), various specs & consumption curves
- ports.csv — 60 global ports with region, coordinates, and ECA flag
- routes.csv — 600+ routes with variants (SUEZ, PANAMA, CAPE, DIRECT), tolls, delays, piracy risk
- bunker_prices.csv — 18 bunker hubs with 120 days of VLSFO & LSMGO price data (random-walk time series)
- port_costs.csv — PDA components (port dues, pilotage, towage, agency) for each port
- canal_fees.csv — reference canal tolls for SUEZ, PANAMA, and CAPE with vessel class
- cargos.csv — 300 cargo posts with quantity, laycan, and terms
- cp_clauses_lib.txt — 5 CP clause templates with laytime and demurrage rules
- laytime_rates.csv — baseline allowed hours + demurrage/despatch by commodity type
- indices.csv — 180 days of BDI & VLSFO index data
- ais_tracks.csv — sparse AIS-like tracks for 30 vessels (hourly data over 7 days)
- README_FINAL_SYNTHETIC.txt — instructions and overview

Instructions:
- Load the CSVs into your preferred environment for processing and modeling.
- The dataset includes detailed maritime parameters for optimization, cost modeling, and scenario analysis.
