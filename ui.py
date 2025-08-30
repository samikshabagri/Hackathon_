import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Maritime AI Agent",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

def make_api_request(endpoint, method="GET", data=None):
    """Make API request to the backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def main():
    # Sidebar
    st.sidebar.title("🚢 Enhanced Maritime AI Agent")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🚀 Enhanced RAG AI Chat", "🗺️ Voyage Planning", "📦 Cargo Matching", "📊 Market Analysis", "💰 PDA Calculator", "🔍 Data Explorer"]
    )
    
    # Main content
    if page == "🚀 Enhanced RAG AI Chat":
        chat_interface()
    elif page == "🗺️ Voyage Planning":
        voyage_planning()
    elif page == "📦 Cargo Matching":
        cargo_matching()
    elif page == "📊 Market Analysis":
        market_analysis()
    elif page == "💰 PDA Calculator":
        pda_calculator()
    elif page == "🔍 Data Explorer":
        data_explorer()

def chat_interface():
    st.title("🚀 Enhanced RAG-Powered Maritime AI Chat")
    st.markdown("""
    **Ultra-Intelligent AI Assistant with Advanced RAG & GPT Integration:**
    - 🧠 **GPT-4 Powered**: Advanced language understanding and generation
    - 🔍 **RAG Retrieval**: Intelligent context retrieval from maritime knowledge base
    - 💡 **Smart Intent Classification**: Automatically understands any type of query
    - 📊 **Comprehensive Analysis**: Executes actions and provides detailed results
    - 🎯 **Few-Shot Learning**: Learns from examples to better understand your needs
    - 🌐 **Natural Language**: Ask questions in any way you want!
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional data if available
            if message.get("data"):
                with st.expander("🔍 View Detailed Data"):
                    st.json(message["data"])
            
            # Show suggestions if available
            if message.get("suggestions"):
                st.markdown("**💡 Suggestions:**")
                for suggestion in message["suggestions"]:
                    st.markdown(f"• {suggestion}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about maritime operations in any way you want..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing with Enhanced RAG & GPT..."):
                response = make_api_request("/chat", method="POST", data={"query": prompt})
                
                if response and response.get("success"):
                    # Display main response
                    st.markdown(response["response"])
                    
                    # Show action results if available
                    if response.get("action_result", {}).get("status") == "success":
                        action_data = response["action_result"]["data"]
                        st.success(f"✅ **Action Completed**: {response['action_result']['action'].replace('_', ' ').title()}")
                        
                        # Display action-specific data
                        if response["action_result"]["action"] == "voyage_planning":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Distance", f"{action_data.get('distance_nm', 0):.0f} NM")
                                st.metric("Duration", f"{action_data.get('total_voyage_days', 0):.1f} days")
                            with col2:
                                st.metric("Total Cost", f"${action_data.get('total_voyage_cost_usd', 0):,.0f}")
                                st.metric("Fuel Cost", f"${action_data.get('fuel_cost_usd', 0):,.0f}")
                            with col3:
                                st.metric("ETA", action_data.get('eta', 'N/A')[:10] if action_data.get('eta') else 'N/A')
                                st.metric("Risk", action_data.get('piracy_risk', 'N/A').title())
                        
                        elif response["action_result"]["action"] == "cargo_matching":
                            matches = action_data.get("matches", [])
                            if matches:
                                st.metric("Matches Found", len(matches))
                                # Show top matches
                                if len(matches) > 0:
                                    top_match = matches[0]
                                    st.info(f"**Top Match**: {top_match.get('cargo_id', 'N/A')} - {top_match.get('commodity', 'N/A')} ({top_match.get('quantity_mt', 0):,.0f} MT)")
                        
                        elif response["action_result"]["action"] == "market_analysis":
                            st.metric("BDI", action_data.get('current_bdi', 0))
                            st.metric("VLSFO Price", f"${action_data.get('current_vlsfo_usd_per_mt', 0)}/MT")
                    
                    # Show suggestions
                    if response.get("suggestions"):
                        st.markdown("**💡 Try These Queries:**")
                        for suggestion in response["suggestions"]:
                            st.markdown(f"• {suggestion}")
                    
                    # Store response with additional data
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "data": response.get("action_result", {}).get("data"),
                        "suggestions": response.get("suggestions")
                    }
                    
                else:
                    error_msg = response.get("response", "I couldn't process that request. Please try again.")
                    st.markdown(error_msg)
                    assistant_message = {"role": "assistant", "content": error_msg}
        
        # Add assistant response to chat history
        st.session_state.messages.append(assistant_message)
    
    # Sidebar for quick actions and examples
    with st.sidebar:
        st.markdown("### 🚀 Enhanced AI Examples")
        
        example_queries = [
            "Plan voyage for vessel 9700001 from BRSSZ to CNSHA",
            "Find cargo matches for Panamax vessels",
            "Show market trends for Capesize",
            "Bunker prices in Singapore",
            "Calculate PDA for vessel 9700001",
            "Compare Suez vs Cape routes",
            "Get vessel information for 9700001",
            "What's the current BDI trend?",
            "Which ports have the cheapest bunker prices?",
            "How do I optimize voyage costs?",
            "What are the best cargo opportunities right now?",
            "Tell me about vessel 9700001",
            "Show me coal cargoes from Australia",
            "What's the freight rate for Panamax Brazil-China?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📋 Get Available Commands"):
            commands = make_api_request("/utils/commands")
            if commands:
                st.sidebar.markdown("**Available Commands:**")
                for cmd in commands.get("commands", []):
                    st.sidebar.text(cmd)
        
        st.markdown("---")
        st.markdown("### 🧠 Enhanced AI Features")
        st.markdown("""
        - **GPT-4 Integration**
        - **RAG Retrieval**
        - **Smart Intent Classification**
        - **Natural Language Understanding**
        - **Context-Aware Responses**
        - **Few-Shot Learning**
        - **Action Execution**
        - **Comprehensive Analysis**
        """)

def voyage_planning():
    st.title("🗺️ Voyage Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Plan Voyage")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        load_port = st.text_input("Load Port", value="BRSSZ")
        disch_port = st.text_input("Discharge Port", value="CNSHA")
        speed_knots = st.slider("Speed (knots)", 10.0, 18.0, 14.0, 0.5)
        route_variant = st.selectbox("Route Variant", ["DIRECT", "SUEZ", "PANAMA", "CAPE"])
        
        if st.button("Plan Voyage"):
            with st.spinner("Planning voyage..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "load_port": load_port,
                    "disch_port": disch_port,
                    "speed_knots": speed_knots,
                    "route_variant": route_variant
                }
                
                result = make_api_request("/voyage/plan", method="POST", data=data)
                
                if result:
                    st.success("Voyage planned successfully!")
                    
                    # Display voyage details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Distance", f"{result['distance_nm']:.0f} NM")
                        st.metric("Duration", f"{result['total_voyage_days']:.1f} days")
                    with col2:
                        st.metric("Total Cost", f"${result['total_voyage_cost_usd']:,.0f}")
                        st.metric("Fuel Cost", f"${result['fuel_cost_usd']:,.0f}")
                    with col3:
                        st.metric("ETA", result['eta'][:10])
                        st.metric("Risk", result['piracy_risk'].title())
                    
                    # Show detailed breakdown
                    with st.expander("Voyage Details"):
                        st.json(result)
    
    with col2:
        st.subheader("Compare Routes")
        
        comp_vessel_imo = st.text_input("Vessel IMO (Compare)", value="9700001")
        comp_load_port = st.text_input("Load Port (Compare)", value="BRSSZ")
        comp_disch_port = st.text_input("Discharge Port (Compare)", value="CNSHA")
        
        if st.button("Compare Routes"):
            with st.spinner("Comparing routes..."):
                data = {
                    "vessel_imo": comp_vessel_imo,
                    "load_port": comp_load_port,
                    "disch_port": comp_disch_port
                }
                
                result = make_api_request("/voyage/compare-routes", method="POST", data=data)
                
                if result and result.get("comparisons"):
                    comparisons = result["comparisons"]
                    
                    # Create comparison table
                    df = pd.DataFrame(comparisons[:5])  # Top 5
                    df = df[['route_variant', 'speed_knots', 'total_voyage_cost_usd', 'total_voyage_days', 'distance_nm']]
                    df.columns = ['Route', 'Speed (kts)', 'Cost (USD)', 'Days', 'Distance (NM)']
                    df['Cost (USD)'] = df['Cost (USD)'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cost comparison chart
                    fig = px.bar(
                        comparisons[:5],
                        x='route_variant',
                        y='total_voyage_cost_usd',
                        title="Route Cost Comparison",
                        labels={'route_variant': 'Route', 'total_voyage_cost_usd': 'Cost (USD)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

def cargo_matching():
    st.title("📦 Cargo Matching")
    
    tab1, tab2, tab3 = st.tabs(["🚢 Find Cargo for Vessel", "📦 Find Vessel for Cargo", "🎯 Optimal Matches"])
    
    with tab1:
        st.subheader("Find Cargo Matches for Vessel")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0)
        max_ballast = st.number_input("Max Ballast Distance (NM)", value=2000.0)
        
        if st.button("Find Cargo Matches"):
            with st.spinner("Finding cargo matches..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "min_tce_usd_per_day": min_tce,
                    "max_ballast_distance_nm": max_ballast
                }
                
                result = make_api_request("/cargo/find-matches", method="POST", data=data)
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])  # Top 10
                        df = df[['cargo_id', 'commodity', 'quantity_mt', 'load_port', 'disch_port', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['cargo_id', 'commodity', 'quantity_mt', 'load_port', 'disch_port', 'TCE (USD/day)']]
                        df.columns = ['Cargo ID', 'Commodity', 'Quantity (MT)', 'Load Port', 'Disch Port', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        
                        # TCE distribution chart
                        tce_values = [m['tce_analysis']['tce_usd_per_day'] for m in matches[:10]]
                        fig = px.histogram(x=tce_values, title="TCE Distribution", labels={'x': 'TCE (USD/day)', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No suitable cargoes found.")
    
    with tab2:
        st.subheader("Find Vessel Matches for Cargo")
        
        cargo_id = st.text_input("Cargo ID", value="CARG-001")
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0, key="vessel_tce")
        
        if st.button("Find Vessel Matches"):
            with st.spinner("Finding vessel matches..."):
                data = {
                    "cargo_id": cargo_id,
                    "min_tce_usd_per_day": min_tce
                }
                
                result = make_api_request("/cargo/find-vessel-matches", method="POST", data=data)
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])
                        df = df[['vessel_name', 'vessel_type', 'dwt', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['vessel_name', 'vessel_type', 'dwt', 'TCE (USD/day)']]
                        df.columns = ['Vessel Name', 'Type', 'DWT', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        st.success(f"✅ Found {len(matches)} vessel matches")
                    else:
                        st.warning("No suitable vessels found.")
                else:
                    st.error("❌ Error: Unable to fetch vessel matches")
                    if result:
                        st.json(result)
    
    with tab3:
        st.subheader("Optimal Vessel-Cargo Combinations")
        
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0, key="optimal_tce")
        max_matches = st.number_input("Max Matches", value=20, min_value=1, max_value=50)
        
        if st.button("Find Optimal Matches"):
            with st.spinner("Finding optimal matches..."):
                result = make_api_request(f"/cargo/optimal-matches?min_tce_usd_per_day={min_tce}&max_matches={max_matches}")
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])
                        df = df[['vessel_name', 'cargo_id', 'commodity', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['vessel_name', 'cargo_id', 'commodity', 'TCE (USD/day)']]
                        df.columns = ['Vessel', 'Cargo', 'Commodity', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        st.success(f"✅ Found {len(matches)} optimal matches")
                    else:
                        st.warning("No optimal matches found.")
                else:
                    st.error("❌ Error: Unable to fetch optimal matches")
                    if result:
                        st.json(result)

def market_analysis():
    st.title("📊 Market Analysis")
    
    # Market summary
    st.subheader("Market Summary")
    
    if st.button("Refresh Market Data"):
        with st.spinner("Loading market data..."):
            summary = make_api_request("/market/summary")
            
            if summary:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("BDI", summary['current_bdi'])
                with col2:
                    st.metric("VLSFO Price", f"${summary['current_vlsfo_usd_per_mt']}/MT")
                with col3:
                    st.metric("Market Sentiment", summary['market_sentiment'].title())
                with col4:
                    st.metric("BDI Trend", f"{summary['bdi_trend_percent']}%")
                
                # Vessel analysis
                st.subheader("Vessel Type Analysis")
                vessel_data = summary.get('vessel_analysis', {})
                
                if vessel_data:
                    vessel_df = pd.DataFrame([
                        {
                            'Vessel Type': k,
                            'Estimated Rate': v['estimated_rate_usd_per_mt'],
                            'Outlook': v['market_outlook']
                        }
                        for k, v in vessel_data.items()
                    ])
                    
                    st.dataframe(vessel_df, use_container_width=True)
    
    # Freight rate trends
    st.subheader("Freight Rate Trends")
    
    vessel_type = st.selectbox("Vessel Type", ["Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"])
    route = st.text_input("Route", value="BRSSZ-CNSHA")
    
    if st.button("Get Trends"):
        with st.spinner("Loading trends..."):
            trends = make_api_request(f"/market/trends/{vessel_type}?route={route}")
            
            if trends:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current BDI", trends['current_bdi'])
                with col2:
                    st.metric("Estimated Rate", f"${trends['estimated_freight_rate_usd_per_mt']}/MT")
                with col3:
                    st.metric("Trend", trends['trend_direction'].title())
                
                st.info(trends['recommendation'])
                
                # Historical data chart
                if trends.get('historical_data'):
                    hist_df = pd.DataFrame(trends['historical_data'])
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    
                    fig = px.line(hist_df, x='date', y='BDI', title=f"{vessel_type} BDI Trend")
                    st.plotly_chart(fig, use_container_width=True)

def pda_calculator():
    st.title("💰 PDA Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calculate PDA")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        load_port = st.text_input("Load Port", value="BRSSZ")
        disch_port = st.text_input("Discharge Port", value="CNSHA")
        bunker_port = st.text_input("Bunker Port (optional)", value="")
        fuel_type = st.selectbox("Fuel Type", ["VLSFO", "HSFO", "LSMGO"])
        
        if st.button("Calculate PDA"):
            with st.spinner("Calculating PDA..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "load_port": load_port,
                    "disch_port": disch_port,
                    "fuel_type": fuel_type
                }
                
                if bunker_port:
                    data["bunker_port"] = bunker_port
                
                result = make_api_request("/pda/calculate", method="POST", data=data)
                
                if result:
                    st.success("PDA calculated successfully!")
                    
                    # Display PDA breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total PDA", f"${result['total_pda_usd']:,.0f}")
                        st.metric("Load Port Fees", f"${result['load_port_fees']['total']:,.0f}")
                        st.metric("Discharge Port Fees", f"${result['disch_port_fees']['total']:,.0f}")
                    with col2:
                        st.metric("Bunker Costs", f"${result['bunker_costs']['total_cost']:,.0f}")
                        st.metric("Canal Costs", f"${result['canal_costs']['canal_toll_usd']:,.0f}")
                        st.metric("Budget Status", result['budget_analysis']['status'])
                    
                    # Cost breakdown pie chart
                    cost_data = {
                        'Port Fees': result['load_port_fees']['total'] + result['disch_port_fees']['total'],
                        'Bunker': result['bunker_costs']['total_cost'],
                        'Canal': result['canal_costs']['canal_toll_usd'],
                        'Additional': result['additional_costs']['total']
                    }
                    
                    fig = px.pie(
                        values=list(cost_data.values()),
                        names=list(cost_data.keys()),
                        title="PDA Cost Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bunker Port Comparison")
        
        comp_vessel_imo = st.text_input("Vessel IMO (Compare)", value="9700001")
        comp_load_port = st.text_input("Load Port (Compare)", value="BRSSZ")
        comp_disch_port = st.text_input("Discharge Port (Compare)", value="CNSHA")
        candidate_ports = st.text_input("Candidate Ports (comma-separated)", value="BRSSZ,SGSIN,AEJEA")
        
        if st.button("Compare Bunker Ports"):
            with st.spinner("Comparing bunker ports..."):
                ports_list = [p.strip() for p in candidate_ports.split(",")]
                
                data = {
                    "vessel_imo": comp_vessel_imo,
                    "load_port": comp_load_port,
                    "disch_port": comp_disch_port,
                    "fuel_type": "VLSFO",
                    "candidate_ports": ports_list
                }
                
                result = make_api_request("/pda/compare-bunker-ports", method="POST", data=data)
                
                if result and result.get("comparisons"):
                    comparisons = result["comparisons"]
                    
                    # Create comparison table
                    df = pd.DataFrame(comparisons)
                    df = df[['port', 'price_usd_per_mt', 'total_cost_usd', 'savings_vs_load_port']]
                    df.columns = ['Port', 'Price (USD/MT)', 'Total Cost (USD)', 'Savings vs Load Port (USD)']
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cost comparison chart
                    fig = px.bar(
                        df,
                        x='Port',
                        y='Total Cost (USD)',
                        title="Bunker Port Cost Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def data_explorer():
    st.title("🔍 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚢 Vessels", "📦 Cargos", "🏢 Ports", "Market Data"])
    
    with tab1:
        st.subheader("Vessel Data")
        
        vessel_type = st.selectbox("Vessel Type", ["", "Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"])
        min_dwt = st.number_input("Min DWT", value=0.0)
        max_dwt = st.number_input("Max DWT", value=200000.0)
        
        if st.button("Load Vessels"):
            with st.spinner("Loading vessel data..."):
                params = f"?min_dwt={min_dwt}&max_dwt={max_dwt}"
                if vessel_type:
                    params += f"&vessel_type={vessel_type}"
                
                result = make_api_request(f"/data/vessels{params}")
                
                if result and result.get("vessels"):
                    vessels = result["vessels"]
                    df = pd.DataFrame(vessels)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Vessel type distribution
                    if 'type' in df.columns:
                        fig = px.pie(df, names='type', title="Vessel Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Cargo Data")
        
        cargo_type = st.selectbox("Cargo Type", ["", "Iron Ore", "Coal", "Grain", "Soybeans", "Sugar", "Fertilizer", "Cement"])
        load_port = st.text_input("Load Port Filter", value="")
        min_qty = st.number_input("Min Quantity (MT)", value=0.0)
        max_qty = st.number_input("Max Quantity (MT)", value=200000.0)
        
        if st.button("Load Cargos"):
            with st.spinner("Loading cargo data..."):
                params = f"?min_quantity={min_qty}&max_quantity={max_qty}"
                if cargo_type:
                    params += f"&cargo_type={cargo_type}"
                if load_port:
                    params += f"&load_port={load_port}"
                
                result = make_api_request(f"/data/cargos{params}")
                
                if result and result.get("cargos"):
                    cargos = result["cargos"]
                    df = pd.DataFrame(cargos)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cargo type distribution
                    if 'commodity' in df.columns:
                        fig = px.pie(df, names='commodity', title="Cargo Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Port Information")
        
        port_code = st.text_input("Port Code", value="BRSSZ")
        
        if st.button("Get Port Details"):
            with st.spinner("Loading port details..."):
                result = make_api_request(f"/data/ports/{port_code}")
                
                if result:
                    st.json(result)
    
    with tab4:
        st.subheader("Market Data")
        
        if st.button("Load Market Summary"):
            with st.spinner("Loading market data..."):
                summary = make_api_request("/utils/summary")
                
                if summary:
                    st.json(summary)

if __name__ == "__main__":
    main()
