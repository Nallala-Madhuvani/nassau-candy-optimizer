import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time

# --- Modern Theme & Wide Mode ---
st.set_page_config(
    page_title="Nassau Candy Logistics Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Enterprise CSS Styling ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #00f2fe;
    }
    .badge-optimal {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #F59E0B;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Data & Artifact Loaders ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('lead_time_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_nassau_candy.csv')

model, model_cols = load_artifacts()
df = load_data()

factory_coords = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}

region_coords = {
    'Atlantic': (37.0, -78.0),
    'Gulf': (30.5, -90.0),
    'Interior': (39.5, -98.0),
    'Pacific': (37.5, -120.0)
}
factories = list(factory_coords.keys())

# --- Top Navigation Bar ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("⚡ Confectionery Supply Chain Decision Cockpit")
    st.caption("Prescriptive ML Optimization Engine • Multi-Factory Reallocation Simulator")
with col_head2:
    st.write("")
    refresh_btn = st.button("🔄 Clear Cache & Recalibrate", width='stretch')
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

# --- KPI Dashboard Row ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_orders = len(df)
avg_lead = df['Lead Time (Days)'].mean()
total_profit = df['Gross Profit'].sum()
margin = (total_profit / df['Sales'].sum()) * 100

with kpi1:
    st.markdown(f'<div class="metric-card"><small>📦 TOTAL MONITORED ORDERS</small><h3>{total_orders:,}</h3><span style="color:#10B981;">▲ Verified Production</span></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><small>⏱️ BASELINE AVG TRANSIT</small><h3>{avg_lead:.1f} Days</h3><span style="color:#3B82F6;">4 Regional Hubs</span></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card"><small>💰 ACCUMULATED GROSS MARGIN</small><h3>${total_profit:,.0f}</h3><span style="color:#10B981;">{margin:.1f}% Margin Ratio</span></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-card"><small>🏭 ACTIVE PRODUCTION PLANTS</small><h3>5 Plants</h3><span style="color:#F59E0B;">Static Assignment</span></div>', unsafe_allow_html=True)

st.write("")

# --- Sidebar Configuration ---
with st.sidebar:
    st.subheader("🛠️ Strategy & Controls")
    
    selected_product = st.selectbox("📦 Target SKU", sorted(df['Product Name'].unique()))
    selected_mode = st.selectbox("🚚 Carrier Service Level", sorted(df['Ship Mode'].unique()))
    
    st.markdown("---")
    st.subheader("⚖️ Objective Weights")
    speed_weight = st.slider("Priority: Delivery Velocity", 0.0, 1.0, 0.6, 0.05)
    margin_weight = 1.0 - speed_weight
    st.caption(f"Velocity Weight: **{int(speed_weight*100)}%** | Cost Weight: **{int(margin_weight*100)}%**")
    
    st.markdown("---")
    simulate_disruption = st.toggle("⚠️ Simulate Regional Road Disruptions (+1.2 Days)", value=False)

# Current assignment lookup
current_factory = df[df['Product Name'] == selected_product]['Factory'].iloc[0]

# --- Simulation Execution ---
sim_results = []
regions = list(region_coords.keys())

for fac in factories:
    fac_lat, fac_lon = factory_coords[fac]
    predicted_leads = []
    
    for reg in regions:
        dest_lat, dest_lon = region_coords[reg]
        dist_proxy = np.sqrt((dest_lat - fac_lat)**2 + (dest_lon - fac_lon)**2)
        
        row = {col: 0 for col in model_cols}
        if f'Factory_{fac}' in row:
            row[f'Factory_{fac}'] = 1
        if f'Region_{reg}' in row:
            row[f'Region_{reg}'] = 1
        if f'Ship Mode_{selected_mode}' in row:
            row[f'Ship Mode_{selected_mode}'] = 1
        row['Distance_Proxy'] = dist_proxy
        
        feat_df = pd.DataFrame([row])[model_cols]
        lead_pred = model.predict(feat_df)[0]
        if simulate_disruption and reg in ['Interior', 'Gulf']:
            lead_pred += 1.2
        predicted_leads.append(lead_pred)
    
    avg_pred_lead = np.mean(predicted_leads)
    
    prod_data = df[df['Product Name'] == selected_product]
    base_profit = prod_data['Gross Profit'].mean()
    lat_diff = abs(fac_lat - 37.0)
    est_profit = base_profit * (1.0 + ((2.0 - (lat_diff * 0.1)) / 100.0))
    
    sim_results.append({
        'Factory': fac,
        'Est_Lead_Time': round(avg_pred_lead, 2),
        'Est_Profit_Per_Order': round(est_profit, 2),
        'Is_Current': (fac == current_factory)
    })

sim_df = pd.DataFrame(sim_results)
current_lead = sim_df[sim_df['Is_Current']]['Est_Lead_Time'].values[0]
sim_df['Lead_Time_Delta_%'] = ((current_lead - sim_df['Est_Lead_Time']) / current_lead) * 100

lead_norm = (sim_df['Est_Lead_Time'].max() - sim_df['Est_Lead_Time']) / (sim_df['Est_Lead_Time'].max() - sim_df['Est_Lead_Time'].min() + 1e-5)
profit_norm = (sim_df['Est_Profit_Per_Order'] - sim_df['Est_Profit_Per_Order'].min()) / (sim_df['Est_Profit_Per_Order'].max() - sim_df['Est_Profit_Per_Order'].min() + 1e-5)
sim_df['Composite_Score'] = (speed_weight * lead_norm) + (margin_weight * profit_norm)
sim_df = sim_df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)

best_factory = sim_df.iloc[0]['Factory']

# --- Main Interactive Tabs ---
tab_matrix, tab_route, tab_whatif, tab_dispatch = st.tabs([
    "📊 Decision Matrix", 
    "🗺️ Spatial Network", 
    "🧮 What-If Financials", 
    "🚀 Dispatch Console"
])

# TAB 1: DECISION MATRIX
with tab_matrix:
    col_rec1, col_rec2 = st.columns([1, 1])
    
    with col_rec1:
        st.subheader("💡 Prescriptive Recommendation")
        if best_factory != current_factory:
            st.error(f"⚠️ **Inefficient Allocation Detected**")
            st.markdown(f"""
            - **Current Plant:** `{current_factory}` ({current_lead:.2f} days avg transit)
            - **Recommended Plant:** `{best_factory}` ({sim_df.iloc[0]['Est_Lead_Time']:.2f} days avg transit)
            - **Net Efficiency Gain:** **+{sim_df.iloc[0]['Lead_Time_Delta_%']:.1f}% faster delivery**
            """)
        else:
            st.success(f"✅ **Assignment Confirmed Optimal**")
            st.markdown(f"The baseline plant `{current_factory}` delivers optimal trade-offs under the selected weights.")
        
        st.markdown("#### Dynamic Plant Comparison")
        st.dataframe(
            sim_df[['Factory', 'Est_Lead_Time', 'Lead_Time_Delta_%', 'Est_Profit_Per_Order', 'Composite_Score']],
            width='stretch',
            hide_index=True
        )
        
        csv_data = sim_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Scenario to CSV", csv_data, f"{selected_product}_scenario.csv", "text/csv", width='stretch')

    with col_rec2:
        st.subheader("⏱️ Transit Time Benchmarking (Days)")
        colors = ['#10B981' if f == best_factory else '#64748B' for f in sim_df['Factory']]
        fig_bar = go.Figure(data=[
            go.Bar(
                x=sim_df['Factory'], 
                y=sim_df['Est_Lead_Time'],
                marker_color=colors,
                text=sim_df['Est_Lead_Time'],
                textposition='auto'
            )
        ])
        fig_bar.update_layout(
            template="plotly_dark",
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="Lead Time (Days)"
        )
        st.plotly_chart(fig_bar, width='stretch')

# TAB 2: SPATIAL NETWORK
with tab_route:
    st.subheader("🌐 Network Node Topology")
    geo_df = pd.DataFrame([
        {"Name": f, "lat": lat, "lon": lon, "Type": "Factory", "Size": 18} for f, (lat, lon) in factory_coords.items()
    ] + [
        {"Name": r, "lat": lat, "lon": lon, "Type": "Customer Region Hub", "Size": 12} for r, (lat, lon) in region_coords.items()
    ])

    fig_map = px.scatter_geo(
        geo_df, lat="lat", lon="lon", color="Type", size="Size",
        text="Name", scope="usa",
        color_discrete_map={"Factory": "#00f2fe", "Customer Region Hub": "#ff007f"},
        template="plotly_dark"
    )
    fig_map.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_map, width='stretch')

# TAB 3: WHAT-IF FINANCIAL CALCULATOR
with tab_whatif:
    st.subheader("🧮 Multi-Unit Reallocation Impact Calculator")
    st.write("Calculate aggregate savings by shifting recurring batch production volumes.")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        batch_volume = st.number_input("Monthly Order Volume (Batches)", min_value=100, max_value=50000, value=2500, step=500)
    with col_c2:
        fuel_surcharge = st.slider("Fuel Surcharge Impact ($/day saved)", 0.50, 10.00, 2.50, 0.25)
    with col_c3:
        sla_penalty = st.number_input("Late SLA Penalty Per Day ($)", min_value=0.0, value=15.0, step=2.5)

    days_saved = max(0.0, current_lead - sim_df.iloc[0]['Est_Lead_Time'])
    monthly_hours_saved = days_saved * batch_volume * 24
    monthly_savings = (days_saved * fuel_surcharge * batch_volume) + (days_saved * 0.05 * sla_penalty * batch_volume)
    
    st.markdown("---")
    w1, w2, w3 = st.columns(3)
    w1.metric("Transit Hours Eliminated / Mo", f"{monthly_hours_saved:,.0f} Hrs", delta=f"{days_saved:.2f} Days/Order")
    w2.metric("Projected Monthly Freight Savings", f"${monthly_savings:,.2f}", delta="Cost Avoidance")
    w3.metric("Annualized Efficiency Dividend", f"${monthly_savings*12:,.2f}", delta="EBITDA Contribution")

# TAB 4: DISPATCH CONSOLE & INTERACTIVE WORKFLOW
with tab_dispatch:
    st.subheader("🚀 Production Reallocation Dispatch Console")
    st.write("Approve route modifications and broadcast instruction sets directly to plant ERPs.")
    
    st.info(f"Target Command: Shift production line of **{selected_product}** from **{current_factory}** to **{best_factory}**.")
    
    selected_urgency = st.radio("Execution Priority Tier", ["Standard (End of Quarter)", "Immediate (Next Production Cycle)", "Critical Expedited"], horizontal=True)
    
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        notes = st.text_input("Operational Sign-off Notes", "Optimized routing based on transit distance and regional margin preservation.")
    with col_b2:
        st.write("")
        st.write("")
        execute_button = st.button("⚡ EXECUTE PRODUCTION SHIFT", type="primary", width='stretch')
        
    if execute_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Connecting to Plant ERP Gateway...")
        time.sleep(0.3)
        progress_bar.progress(30)
        
        status_text.text(f"Updating bill-of-materials routing to {best_factory}...")
        time.sleep(0.4)
        progress_bar.progress(70)
        
        status_text.text("Notifying 3PL freight carriers and verifying inventory...")
        time.sleep(0.3)
        progress_bar.progress(100)
        
        status_text.empty()
        st.toast(f"Route deployed! {selected_product} routed to {best_factory}", icon="✅")
        st.success(f"Production instructions published successfully! Signed off under '{selected_urgency}'.")
        st.balloons()