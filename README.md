# 🚀 Automated Data Audit & Reporting Engine

Automated performance monitoring and validation tool for high-velocity datasets. Developed to replace manual spreadsheet reporting with real-time, data-driven backend insights.

## 📊 Overview
In data-heavy environments, manual reporting is prone to errors and delays. This system automates the entire analysis pipeline:

- **Ingests Data:** Pulls raw datasets directly from external APIs and Google Sheets endpoints.
- **Analyzes Performance:** Calculates throughput, conversion rates, and custom metrics in real-time.
- **Detects Anomalies:** Identifies statistical outliers and failing data nodes.
- **Actionable Alerts:** Sends a daily executive summary and instant critical alerts to external Webhooks / Messaging APIs.

## 🛠 Tech Stack
- **Python 3.9+** (Core Logic)
- **Pandas** (Data Processing, Vectorized Calculations, and Normalization)
- **Google Sheets API** (Data Ingestion)
- **Messaging APIs** (Real-time Notification System)
- **Git** (Version Control)

## 💡 Key Features

### 1. Automated Decision Engine
The script applies complex business logic to evaluate raw data:
- 🚀 **SCALING:** High performance metrics. Recommendation: Increase processing allocation.
- 🛡️ **OPTIMIZATION:** Normal operation. Recommendation: Standard monitoring.
- ❌ **KILL SWITCH:** High error rate detected. Recommendation: Halt ingestion immediately.

### 2. Instant Routing Alerts
No need to open dashboards. The engine pushes insights directly to your integrated channels:
* *"🚨 PIPELINE BLEED: Node 6 is failing (Error 0.54). Halt ingestion!"*
* *"🦄 METRIC ALERT: Stream 3 achieved 100% processing rate!"*

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/yevhens-hue/marketing-audit.git
cd marketing-audit
```

2. **Run the Audit Bot:**
```bash
python3 audit_bot.py
```
