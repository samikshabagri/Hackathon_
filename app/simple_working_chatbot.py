import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Local imports
from .data_loader import DataLoader
from .voyage_planner import VoyagePlanner
from .cargo_matcher import CargoMatcher
from .market_insights import MarketInsights
from .pda_calculator import PDACalculator

class SimpleWorkingChatbot:
    """
    Simple working maritime chatbot without external API dependencies.
    Handles basic maritime queries and provides intelligent responses.
    """
    
    def __init__(self):
        # Initialize components
        self.data = DataLoader()
        self.voyage_planner = VoyagePlanner()
        self.cargo_matcher = CargoMatcher()
        self.market_insights = MarketInsights()
        self.pda_calculator = PDACalculator()
        
        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Conversation context
        self.conversation_history = []
        
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize comprehensive maritime knowledge base."""
        return {
            "vessel_types": ["Panamax", "Capesize", "Handysize", "Supramax", "Kamsarmax"],
            "port_mappings": {
                "santos": "BRSSZ", "qingdao": "CNSHA", "shanghai": "CNSHA",
                "rotterdam": "NLRTM", "singapore": "SGSIN", "fujairah": "AEFJR",
                "houston": "USGDX", "new york": "USNYC", "hamburg": "DEHAM",
                "tianjin": "CNTXG", "dalian": "CNDLC", "ningbo": "CNNGB",
                "guangzhou": "CNGZG", "shenzhen": "CNSZX", "xiamen": "CNXMN"
            },
            "cargo_types": ["Iron Ore", "Coal", "Grain", "Soybeans", "Sugar", "Fertilizer", "Cement"],
            "fuel_types": ["VLSFO", "HSFO", "LSMGO", "MGO"],
            "route_variants": ["DIRECT", "SUEZ", "PANAMA", "CAPE"],
            "common_queries": [
                "voyage planning", "cargo matching", "market analysis", "bunker prices",
                "port costs", "freight rates", "vessel information", "route optimization"
            ]
        }
    
    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to handle user queries with intelligent response generation.
        """
        try:
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "user_query": query,
                "type": "user"
            })
            
            # Step 1: Query understanding and intent classification
            intent, extracted_data = self._classify_intent(query)
            
            # Step 2: Generate intelligent response
            response = self._generate_response(query, intent, extracted_data)
            
            # Step 3: Execute actions if needed
            action_result = self._execute_actions(intent, extracted_data)
            
            # Step 4: Generate final response with suggestions
            final_response = self._generate_final_response(response, action_result, query)
            
            # Store response in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "assistant_response": final_response,
                "type": "assistant",
                "intent": intent,
                "extracted_data": extracted_data
            })
            
            return {
                "success": True,
                "response": final_response,
                "intent": intent,
                "extracted_data": extracted_data,
                "action_result": action_result,
                "suggestions": self._generate_suggestions(intent, extracted_data)
            }
            
        except Exception as e:
            error_response = self._handle_error(query, str(e))
            return {
                "success": False,
                "response": error_response,
                "error": str(e)
            }
    
    def _classify_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Classify query intent using pattern matching."""
        query_lower = query.lower()
        extracted_data = {}
        
        # Extract vessel IMO
        imo_match = re.search(r'(\d{7})', query)
        if imo_match:
            extracted_data["vessel_imo"] = imo_match.group(1)
        
        # Extract port codes (5-letter codes like BRSSZ, CNSHA)
        port_codes = re.findall(r'\b[A-Z]{5}\b', query.upper())
        if len(port_codes) >= 2:
            extracted_data["load_port"] = port_codes[0]
            extracted_data["disch_port"] = port_codes[1]
        elif len(port_codes) == 1:
            if "from" in query_lower:
                extracted_data["load_port"] = port_codes[0]
            elif "to" in query_lower:
                extracted_data["disch_port"] = port_codes[0]
        
        # Extract ports
        for port_name, port_code in self.knowledge_base["port_mappings"].items():
            if port_name in query_lower:
                if "from" in query_lower or "load" in query_lower:
                    extracted_data["load_port"] = port_code
                elif "to" in query_lower or "discharge" in query_lower:
                    extracted_data["disch_port"] = port_code
        
        # Extract vessel types
        for vessel_type in self.knowledge_base["vessel_types"]:
            if vessel_type.lower() in query_lower:
                extracted_data["vessel_type"] = vessel_type
                break
        
        # Extract cargo types
        for cargo_type in self.knowledge_base["cargo_types"]:
            if cargo_type.lower() in query_lower:
                extracted_data["cargo_type"] = cargo_type
                break
        
        # Extract cargo ID (CARG-XXX format)
        cargo_id_match = re.search(r'CARG-\d+', query.upper())
        if cargo_id_match:
            extracted_data["cargo_id"] = cargo_id_match.group(0)
        
        # Determine intent
        if any(word in query_lower for word in ["plan", "voyage", "route"]):
            intent = "voyage_planning"
        elif any(word in query_lower for word in ["optimal", "best"]) and any(word in query_lower for word in ["match", "combination"]):
            intent = "optimal_matches"
        elif any(word in query_lower for word in ["cargo", "match", "find"]) and "vessel" in query_lower and "for cargo" in query_lower:
            intent = "vessel_matching"
        elif any(word in query_lower for word in ["cargo", "match", "find"]):
            intent = "cargo_matching"
        elif any(word in query_lower for word in ["market", "trend", "bdi"]):
            intent = "market_analysis"
        elif any(word in query_lower for word in ["bunker", "fuel"]):
            intent = "bunker_analysis"
        elif any(word in query_lower for word in ["pda", "cost", "disbursement"]):
            intent = "pda_calculation"
        elif any(word in query_lower for word in ["vessel", "ship", "imo"]):
            intent = "vessel_info"
        elif any(word in query_lower for word in ["port", "harbor"]):
            intent = "port_info"
        else:
            intent = "general_inquiry"
        
        return intent, extracted_data
    
    def _generate_response(self, query: str, intent: str, extracted_data: Dict[str, Any]) -> str:
        """Generate intelligent response based on intent and data."""
        
        if intent == "voyage_planning":
            if extracted_data.get("vessel_imo") and extracted_data.get("load_port") and extracted_data.get("disch_port"):
                return f"I'll help you plan a voyage for vessel {extracted_data['vessel_imo']} from {extracted_data['load_port']} to {extracted_data['disch_port']}. Let me calculate the optimal route and costs."
            else:
                return "I can help you plan voyages! Please provide a vessel IMO and load/discharge ports. Example: 'Plan voyage for vessel 9700001 from BRSSZ to CNSHA'"
        
        elif intent == "cargo_matching":
            if extracted_data.get("vessel_imo"):
                return f"I'll find suitable cargo matches for vessel {extracted_data['vessel_imo']}. Let me search for profitable opportunities."
            elif extracted_data.get("vessel_type"):
                return f"I'll find cargo matches for {extracted_data['vessel_type']} vessels. Let me search for compatible cargoes."
            else:
                return "I can help you find cargo matches! Please provide a vessel IMO or vessel type. Example: 'Find cargo for vessel 9700001' or 'Find cargo for Panamax vessels'"
        
        elif intent == "vessel_matching":
            if extracted_data.get("cargo_id"):
                return f"I'll find suitable vessels for cargo {extracted_data['cargo_id']}. Let me search for compatible vessels."
            else:
                return "I can help you find vessels for cargo! Please provide a cargo ID. Example: 'Find vessels for cargo CARG-001'"
        
        elif intent == "optimal_matches":
            return "I'll find the optimal vessel-cargo combinations with the highest profitability. Let me analyze all available matches."
        
        elif intent == "market_analysis":
            if extracted_data.get("vessel_type"):
                return f"I'll analyze market trends for {extracted_data['vessel_type']} vessels, including freight rates and market sentiment."
            else:
                return "I can provide market analysis! Please specify a vessel type. Example: 'Market analysis for Capesize' or 'Show market trends for Panamax'"
        
        elif intent == "bunker_analysis":
            return "I'll analyze bunker prices and fuel costs. Let me check current VLSFO prices and compare bunker ports."
        
        elif intent == "pda_calculation":
            if extracted_data.get("vessel_imo"):
                return f"I'll calculate the Port Disbursement Account (PDA) for vessel {extracted_data['vessel_imo']}, including port fees, bunker costs, and other expenses."
            else:
                return "I can calculate PDA costs! Please provide a vessel IMO. Example: 'Calculate PDA for vessel 9700001'"
        
        elif intent == "vessel_info":
            if extracted_data.get("vessel_imo"):
                return f"I'll provide detailed information about vessel {extracted_data['vessel_imo']}, including specifications and performance data."
            else:
                return "I can provide vessel information! Please provide a vessel IMO. Example: 'Vessel info for 9700001'"
        
        elif intent == "port_info":
            return "I'll provide port information including fees, bunker prices, and facilities."
        
        else:
            return "I'm your maritime AI assistant! I can help with voyage planning, cargo matching, market analysis, bunker costs, PDA calculations, and vessel information. What would you like to know?"
    
    def _execute_actions(self, intent: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions based on intent and extracted data."""
        try:
            if intent == "voyage_planning" and extracted_data.get("vessel_imo") and extracted_data.get("load_port") and extracted_data.get("disch_port"):
                return self._execute_voyage_planning(extracted_data)
            elif intent == "cargo_matching" and extracted_data.get("vessel_imo"):
                return self._execute_cargo_matching(extracted_data)
            elif intent == "vessel_matching" and extracted_data.get("cargo_id"):
                return self._execute_vessel_matching(extracted_data)
            elif intent == "optimal_matches":
                return self._execute_optimal_matches(extracted_data)
            elif intent == "market_analysis":
                return self._execute_market_analysis(extracted_data)
            elif intent == "bunker_analysis":
                return self._execute_bunker_analysis(extracted_data)
            elif intent == "pda_calculation" and extracted_data.get("vessel_imo"):
                return self._execute_pda_calculation(extracted_data)
            elif intent == "vessel_info" and extracted_data.get("vessel_imo"):
                return self._execute_vessel_info(extracted_data)
            else:
                return {"status": "no_action", "message": "Need more information to execute action"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_voyage_planning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute voyage planning action."""
        try:
            voyage_plan = self.voyage_planner.plan_voyage(
                vessel_imo=data["vessel_imo"],
                load_port=data["load_port"],
                disch_port=data["disch_port"],
                speed_knots=14.0
            )
            return {
                "status": "success",
                "action": "voyage_planning",
                "data": voyage_plan
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_cargo_matching(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cargo matching action."""
        try:
            matches = self.cargo_matcher.find_cargo_matches(
                vessel_imo=data["vessel_imo"],
                min_tce_usd_per_day=5000
            )
            return {
                "status": "success",
                "action": "cargo_matching",
                "data": {"matches": matches}  # Wrap in dict to match expected format
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_vessel_matching(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vessel matching action."""
        try:
            matches = self.cargo_matcher.find_vessel_matches(
                cargo_id=data["cargo_id"],
                min_tce_usd_per_day=5000
            )
            return {
                "status": "success",
                "action": "vessel_matching",
                "data": {"matches": matches}  # Wrap in dict to match expected format
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_optimal_matches(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimal matches action."""
        try:
            matches = self.cargo_matcher.get_optimal_matches(
                min_tce_usd_per_day=5000,
                max_matches=20
            )
            return {
                "status": "success",
                "action": "optimal_matches",
                "data": {"matches": matches}  # Wrap in dict to match expected format
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_market_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market analysis action."""
        try:
            summary = self.market_insights.get_market_summary()
            return {
                "status": "success",
                "action": "market_analysis",
                "data": summary
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_bunker_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bunker analysis action."""
        try:
            # Use a default port for bunker analysis
            analysis = self.market_insights.get_bunker_price_analysis(port="SGSIN")
            return {
                "status": "success",
                "action": "bunker_analysis",
                "data": analysis
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_pda_calculation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PDA calculation action."""
        try:
            if not data.get("load_port") or not data.get("disch_port"):
                return {
                    "status": "incomplete",
                    "message": "Need load and discharge ports for PDA calculation"
                }
            
            voyage_plan = self.voyage_planner.plan_voyage(
                vessel_imo=data["vessel_imo"],
                load_port=data["load_port"],
                disch_port=data["disch_port"]
            )
            
            pda = self.pda_calculator.calculate_pda(
                voyage_plan=voyage_plan,
                fuel_type="VLSFO"
            )
            return {
                "status": "success",
                "action": "pda_calculation",
                "data": pda
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_vessel_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vessel information action."""
        try:
            vessel_imo = data["vessel_imo"]
            vessel_data = self.data.vessels[self.data.vessels["imo"] == vessel_imo]
            
            if not vessel_data.empty:
                return {
                    "status": "success",
                    "action": "vessel_info",
                    "data": vessel_data.iloc[0].to_dict()
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Vessel {vessel_imo} not found in database"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _generate_final_response(self, response: str, action_result: Dict[str, Any], original_query: str) -> str:
        """Generate final response with action results and suggestions."""
        final_response = response
        
        # Add action results
        if action_result.get("status") == "success":
            final_response += f"\n\n✅ **Action Completed**: {self._format_action_result(action_result)}"
        elif action_result.get("status") == "incomplete":
            final_response += f"\n\n⚠️ **Additional Information Needed**: {action_result['message']}"
        elif action_result.get("status") == "error":
            final_response += f"\n\n❌ **Error**: {action_result['message']}"
        
        return final_response
    
    def _format_action_result(self, action_result: Dict[str, Any]) -> str:
        """Format action result for display."""
        action = action_result.get("action", "")
        data = action_result.get("data", {})
        
        if action == "voyage_planning":
            return f"Voyage planned successfully! Distance: {data.get('distance_nm', 0):.0f} NM, Cost: ${data.get('total_voyage_cost_usd', 0):,.0f}"
        elif action == "cargo_matching":
            matches = data.get("matches", [])
            return f"Found {len(matches)} cargo matches"
        elif action == "vessel_matching":
            matches = data.get("matches", [])
            return f"Found {len(matches)} vessel matches"
        elif action == "optimal_matches":
            matches = data.get("matches", [])
            return f"Found {len(matches)} optimal vessel-cargo combinations"
        elif action == "market_analysis":
            return f"Market analysis complete. BDI: {data.get('current_bdi', 0)}"
        elif action == "bunker_analysis":
            return f"Bunker analysis complete"
        elif action == "pda_calculation":
            return f"PDA calculated: ${data.get('total_pda_usd', 0):,.0f}"
        elif action == "vessel_info":
            return f"Vessel information retrieved"
        
        return "Action completed successfully"
    
    def _generate_suggestions(self, intent: str, extracted_data: Dict[str, Any]) -> List[str]:
        """Generate intelligent suggestions based on intent and context."""
        suggestions = []
        
        if intent == "voyage_planning":
            if not extracted_data.get("disch_port"):
                suggestions.append("💡 Try: 'Plan voyage for vessel 9700001 from BRSSZ to CNSHA'")
            suggestions.append("💡 Try: 'Compare Suez vs Cape routes for vessel 9700001'")
            suggestions.append("💡 Try: 'Optimize speed for vessel 9700001'")
        
        elif intent == "cargo_matching":
            suggestions.append("💡 Try: 'Find cargo matches for vessel 9700001 with TCE > $5000/day'")
            suggestions.append("💡 Try: 'Find vessels for coal cargo from Australia'")
            suggestions.append("💡 Try: 'Show optimal vessel-cargo combinations'")
        
        elif intent == "market_analysis":
            suggestions.append("💡 Try: 'Show freight rate trends for Panamax vessels'")
            suggestions.append("💡 Try: 'Market opportunities for Capesize vessels'")
            suggestions.append("💡 Try: 'BDI trend analysis'")
        
        elif intent == "bunker_analysis":
            suggestions.append("💡 Try: 'Compare bunker prices in Singapore vs Fujairah'")
            suggestions.append("💡 Try: 'VLSFO price trends'")
            suggestions.append("💡 Try: 'Best bunker ports for vessel 9700001'")
        
        else:
            suggestions.extend([
                "💡 Try: 'Plan voyage for vessel 9700001 from BRSSZ to CNSHA'",
                "💡 Try: 'Find cargo matches for Panamax vessels'",
                "💡 Try: 'Market analysis for Capesize'",
                "💡 Try: 'Bunker prices in Singapore'",
                "💡 Try: 'PDA calculation for vessel 9700001'"
            ])
        
        return suggestions
    
    def _handle_error(self, query: str, error: str) -> str:
        """Handle errors gracefully with helpful suggestions."""
        return f"""I encountered an error processing your query: "{query}"

❌ **Error**: {error}

💡 **Suggestions**:
• Make sure vessel IMO numbers are 7 digits (e.g., 9700001)
• Use standard port codes (e.g., BRSSZ, CNSHA, SGSIN)
• Try simpler queries first, then add more details

🔍 **Available Features**:
• Voyage Planning: "Plan voyage for vessel 9700001 from BRSSZ to CNSHA"
• Cargo Matching: "Find cargo for Panamax vessels"
• Market Analysis: "Show market trends for Capesize"
• Bunker Analysis: "Bunker prices in Singapore"
• PDA Calculation: "Calculate PDA for vessel 9700001"

How can I help you today?"""
    
    def get_available_commands(self) -> List[str]:
        """Get list of available commands and examples."""
        return [
            "Plan voyage for vessel 9700001 from BRSSZ to CNSHA",
            "Find cargo matches for Panamax vessels",
            "Show market trends for Capesize",
            "Compare bunker prices in Singapore vs Fujairah",
            "Calculate PDA for vessel 9700001",
            "Get vessel information for 9700001",
            "Compare Suez vs Cape routes",
            "Find optimal vessel-cargo combinations",
            "What's the current BDI trend?",
            "Which ports have the cheapest bunker prices?",
            "How do I optimize voyage costs?",
            "What are the best cargo opportunities right now?"
        ]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for context."""
        return self.conversation_history
