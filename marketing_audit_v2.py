import pandas as pd
import numpy as np
import os

# --- 1. DATA CLEANING UTILITY ---
def clean_df(df):
    # Strip whitespace and chars
    df.columns = [c.replace('*', '').strip() for c in df.columns]
    
    def clean_numeric(col):
        if col not in df.columns: return pd.Series(0, index=df.index)
        if df[col].dtype == object or str(df[col].dtype).startswith('string'):
            return pd.to_numeric(df[col].astype(str).str.replace(r'[$\s\xa0%]', '', regex=True)
                                     .str.replace(',', '.', regex=False), errors='coerce').fillna(0)
        return df[col].fillna(0).infer_objects(copy=False)

    numeric_cols = ['Costs', 'Visits', 'Regs', 'RFD', 'In', 'Frequency Deposit', '% One timers']
    for col in numeric_cols:
        df[col] = clean_numeric(col)
    
    # Handle technical IDs
    for col in ['Buyer', 'Funnel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str)
            
    df['Cohort_ID'] = df['Buyer'] + "-F" + df['Funnel']
    return df

# --- 2. LOGIC A: CLASSIC ROI ANALYSIS (Old Variant) ---
def analyze_classic(df):
    df_c = df.copy()
    df_c['ROAS'] = np.where(df_c['Costs'] > 0, df_c['In'] / df_c['Costs'], 0)
    df_c['CPA'] = np.where(df_c['RFD'] > 0, df_c['Costs'] / df_c['RFD'], 0)
    
    strategies = []
    for _, row in df_c.iterrows():
        roas, cost = row['ROAS'], row['Costs']
        if roas > 2.0 and cost > 1000: rec = "🚀 SCALE"
        elif roas < 0.6 and cost > 2000: rec = "❌ STOP"
        elif roas > 0.9: rec = "📈 MAINTAIN"
        else: rec = "🛡️ MONITOR"
        strategies.append(rec)
    df_c['Classic_Recommendation'] = strategies
    return df_c

# --- 3. LOGIC B: PREDICTIVE & BENCHMARKING (GoPractice Variant) ---
def analyze_predictive(df):
    df_p = df.copy()
    # ROAS & CPA
    df_p['ROAS'] = np.where(df_p['Costs'] > 0, df_p['In'] / df_p['Costs'], 0)
    df_p['CPA'] = np.where(df_p['RFD'] > 0, df_p['Costs'] / df_p['RFD'], 0)
    df_p['CVR'] = np.where(df_p['Visits'] > 0, (df_p['Regs'] / df_p['Visits']) * 100, 0)
    
    # --- FORECASTING ---
    df_p['Retention_Rate'] = (100 - df_p['% One timers']) / 100
    df_p['Growth_Factor'] = 1 + np.log1p(df_p['Frequency Deposit'] - 1).clip(lower=0) * df_p['Retention_Rate']
    df_p['Projected_ROAS_6M'] = df_p['ROAS'] * df_p['Growth_Factor']
    
    # --- BENCHMARKING (Compare against Buyer Average) ---
    buyer_avgs = df_p.groupby('Buyer').agg({'CPA': 'mean'}).rename(columns={'CPA': 'Buyer_Avg_CPA'})
    df_p = df_p.merge(buyer_avgs, on='Buyer')
    
    def strategic_status(row):
        if row['RFD'] < 5: return '🔬 SMALL SAMPLE'
        if row['Projected_ROAS_6M'] > 1.3: return '💎 FUTURE WINNER'
        if row['ROAS'] < 0.7: return '🛑 BUDGET WASTE'
        return '⚖️ EFFICIENT'
        
    df_p['Predictive_Status'] = df_p.apply(strategic_status, axis=1)
    return df_p

# --- 4. INTEGRATED REPORTING ---
def run_full_audit(csv_path):
    print(f"📦 Loading Data: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df = clean_df(df_raw)
    
    classic = analyze_classic(df)
    predictive = analyze_predictive(df)
    
    print("\n" + "="*80)
    print("🤖 UNIFIED AUDIT REPORT: CLASSIC VS PREDICTIVE LOGIC")
    print("="*80)
    
    # 1. Comparison of top cohorts
    comparison = classic[['Cohort_ID', 'ROAS', 'Classic_Recommendation']].copy()
    comparison = comparison.merge(predictive[['Cohort_ID', 'Projected_ROAS_6M', 'Predictive_Status']], on='Cohort_ID')
    
    print("\n>>> 📈 TOP COMPARISONS (Ranked by Projected Growth):")
    print(comparison.sort_values(by='Projected_ROAS_6M', ascending=False).head(10).to_markdown(index=False))
    
    # 2. Key Insights
    print("\n" + "-"*80)
    print("💡 STRATEGIC INSIGHTS:")
    winners = comparison[comparison['Predictive_Status'] == '💎 FUTURE WINNER']
    waste = comparison[comparison['Predictive_Status'] == '🛑 BUDGET WASTE']
    
    print(f"1. **WINNERS:** Found {len(winners)} cohorts with high LTV potential despite current ROI.")
    print(f"2. **WASTE:** Classic logic suggested keeping {len(waste[waste['Classic_Recommendation'] != '❌ STOP'])} lines that GoPractice labels as Budget Waste.")
    print("-"*80)

if __name__ == "__main__":
    DATA_FILE = '/Users/yevhen/Cursor/Conv TL/marketing_audit_20260208.csv'
    if os.path.exists(DATA_FILE):
        run_full_audit(DATA_FILE)
    else:
        print("Data file not found.")
