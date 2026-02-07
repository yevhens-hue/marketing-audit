# 🚀 Automated Marketing Audit System

**Automated performance monitoring tool for iGaming marketing campaigns.**  
*Developed to replace manual Excel reporting with real-time, data-driven insights.*

---

## 📊 Overview

In high-velocity User Acquisition (UA), manual reporting is too slow. This system automates the entire analysis pipeline:
1.  **Ingests Data:** Pulls raw campaign data (Spend, FTDs, GGR) directly from **Google Sheets** (simulating CRM/Tracker export).
2.  **Analyzes Performance:** Calculates **ROAS**, **CPA**, and **Conversion Rates** in real-time.
3.  **Detects Anomalies:** Identifies "Money Burners" (ROAS < 0.6) and "Hidden Gems" (ROAS > 2.0).
4.  **Actionable Alerts:** Sends a daily executive summary and instant critical alerts to a **Telegram Channel**.

## 🛠 Tech Stack

*   **Python 3.9+** (Core Logic)
*   **Pandas** (Data Processing & Vectorized Calculations)
*   **Google Sheets API** (Data Source)
*   **Telegram Bot API** (Notification System)
*   **Git/GitHub** (Version Control)

## 💡 Key Features

### 1. Automated Decision Engine
The script applies business logic to categorize campaigns:
*   🚀 **SCALING:** High ROAS (>2.0) + Good Volume. Recommendation: *Triple Budget*.
*   🛡️ **OPTIMIZATION:** Good Volume, Marginally Profitable. Recommendation: *Lower CPA*.
*   ❌ **KILL SWITCH:** Low ROAS (<0.6) + High Spend. Recommendation: *Stop Immediately*.

### 2. Instant Telegram Alerts
No need to open dashboards. The bot pushes insights directly to your phone:
> "🚨 **BUDGET BLEED:** Buyer 6 is burning cash (ROAS 0.54). Stop campaign!"
> "🦄 **UNICORN ALERT:** Buyer 3 achieved ROAS 6.34! Scale now!"

### 3. Geo & Product Strategy
Auto-generates a matrix of performance across different Geos and Products (e.g., *Switch budget from Geo 2 to Geo 5*).

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yevhens-hue/marketing-audit.git
    cd marketing-audit
    ```

2.  **Install Dependencies:**
    ```bash
    pip3 install pandas requests
    ```

3.  **Configure:**
    Update `marketing_bot.py` with your Telegram Bot Token and Chat ID.

4.  **Run:**
    ```bash
    python3 marketing_bot.py
    ```

## 📈 Future Roadmap
*   [ ] **Predicitive LTV:** Integrate Machine Learning model to score traffic quality on Day 1.
*   [ ] **Bid Management Hook:** Automatically update bids in Facebook Ads Manager via API.
*   [ ] **Airflow Integration:** Schedule hourly runs via Apache Airflow.

---
*Created by Yevhen Shaforostov as a Proof of Concept for Head of UA / Lead Analyst role.*
