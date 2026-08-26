# 📊 Marketing Audit & Data Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

An automated high-throughput data extraction, marketing audit, and attribution pipeline designed to ingest, clean, and analyze multi-channel advertising performance (Google Ads, Meta Ads, TikTok) and SEO metrics at scale.

---

## 🏛️ Pipeline Architecture

```mermaid
flowchart LR
    subgraph Ingestion ["Multi-Channel Data Ingestion"]
        API_G["Google / Meta Ads API"]
        SEO["SEO & Domain Metrics (Adsy)"]
        CSV["Raw CSV / JSON Exports"]
    end

    subgraph Processing ["Core Processing Engine (Python)"]
        Parser["ETL & Normalization Layer"]
        Anomaly["Anomaly & Fatigue Detector"]
        Scorer["Lead & Attribution Scorer"]
    end

    subgraph Storage ["Storage & Distribution"]
        DB[(PostgreSQL / Supabase)]
        Cache[(Redis Cache)]
        TG["Telegram Alert Bot"]
        Dash["Live Analytics Dashboard"]
    end

    Ingestion --> Parser
    Parser --> Anomaly --> Scorer
    Scorer --> DB
    Scorer --> Cache
    Anomaly -->|Instant Waste Alert| TG
    DB --> Dash
```

---

## ⚡ Core Capabilities

1. **Automated Anomaly & Fatigue Detection:** Flags ROAS drop-offs, ad creative fatigue, and wasted budget segments in real time.
2. **Deterministic Attribution Modeling:** Cleanly tracks user touchpoints across multi-channel campaigns with zero data loss.
3. **High-Speed Data Ingestion:** Asynchronous batch processing handling thousands of daily data points with automated deduplication and schema validation (Pydantic V2).
4. **Instant Alerts & Reporting:** Webhook-driven dispatcher sending formatted executive summaries directly to Telegram and Slack channels.

---

## 🛠️ Tech Stack

- **Core:** Python 3.11, FastAPI, Pydantic V2, Pandas, NumPy
- **Data & Cache:** PostgreSQL, Supabase, Redis
- **Automation & Scheduling:** Celery, Redis Queues, Cron
- **Integrations:** Telegram Bot API, Google Ads API, Meta Marketing API

---

## 👨‍💻 Author & Engineering
- **Author:** [Yevhen Shaforostov](https://github.com/yevhens-hue)
- **Role:** AI Product Manager & Full-Stack AI Engineer at [Adsy.com](https://adsy.com)
