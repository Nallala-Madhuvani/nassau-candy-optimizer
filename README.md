# ⚡ Confectionery Supply Chain Decision Cockpit & Factory Reallocation Engine

A prescriptive decision intelligence and machine learning platform built for **Nassau Candy Distributor** to optimize factory production assignments, minimize transit lead times, and preserve gross profit margins across regional distribution hubs.

---

## 📌 Project Overview
Confectionery distributors traditionally operate under rigid, static SKU-to-plant production assignments. This often produces cross-country transit inefficiencies and elevated carrier lead times. 

This platform replaces static routing heuristics with an end-to-end prescriptive decision engine combining:
* **Predictive ML Modeling:** A Random Forest Regressor predicting order transit lead time based on geographic centroids, transit distance proxies, and carrier service levels.
* **Prescriptive Multi-Objective Optimization:** Dynamic reallocation algorithms balancing speed acceleration vs. regional gross margin preservation.
* **Interactive Operations Dashboard:** An enterprise-grade decision cockpit built in Streamlit featuring interactive what-if financial calculators, geospatial topologies, and an ERP dispatch console.

---

## 📊 Key Operational Metrics
* **Total Transactions Monitored:** 10,194 verified distribution orders
* **Historical Mean Lead Time:** 7.0 Days across standard carrier modes
* **Baseline Gross Profit Pool:** $93,443 (65.9% Gross Margin)
* **Production Facilities:** 5 manufacturing plants (*Lot's O' Nuts, Wicked Choccy's, Sugar Shack, Secret Factory, The Other Factory*)
* **Distribution Centroids:** 4 consumer regions (*Atlantic, Gulf, Interior, Pacific*)

---

## 🧠 Machine Learning Architecture
* **Algorithm:** Random Forest Regressor (`n_estimators=100`)
* **Features:** 
  * `Factory` (One-hot encoded)
  * `Region` (One-hot encoded)
  * `Ship Mode` (One-hot encoded: Same Day, First Class, Second Class, Standard Class)
  * `Distance_Proxy` (Euclidean distance between plant coordinates and regional centroids)
* **Evaluation Metrics:**
  * **Mean Absolute Error (MAE):** `0.24 Days` (~6 hours precision)
  * **R² Score:** `0.9817` (explains >98% of transit variance)

---

## 🚀 Key Dashboard Features
1. **Decision Matrix & Live Reallocation:** Evaluates alternative plants for any selected SKU and automatically flags inefficient assignments with quantified speed improvements (e.g., +7.5% faster delivery).
2. **Spatial Network Topology:** Live interactive map tracking geographic positions of production hubs against regional distribution centers.
3. **What-If Financial Impact Simulator:** Real-time formula estimating monthly hours eliminated, freight fuel cost savings, and late SLA penalties saved across batch production volumes.
4. **Interactive Dispatch Console:** Simulates production line transfer commands, broadcasting routing updates to ERP systems with real-time visual execution.

---

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Nallala-Madhuvani/nassau-candy-optimizer](https://github.com/Nallala-Madhuvani/nassau-candy-optimizer.git)
   cd nassau-candy-optimizer
