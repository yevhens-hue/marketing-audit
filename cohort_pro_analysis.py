import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. DATA LOADING & CLEANING ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = [c.replace('*', '').strip() for c in df.columns]
    
    def clean_numeric(col):
        if col not in df.columns: return pd.Series(0, index=df.index)
        if df[col].dtype == object:
            return pd.to_numeric(df[col].str.replace('$', '', regex=False)
                                     .str.replace('%', '', regex=False)
                                     .str.replace(',', '.', regex=False)
                                     .str.replace(u'\xa0', u'', regex=False)
                                     .str.strip(), errors='coerce').fillna(0)
        return df[col].fillna(0)

    cols_to_clean = ['Costs', 'Visits', 'Regs', 'RFD', 'In', 'Frequency Deposit', '% One timers']
    for col in cols_to_clean:
        df[col] = clean_numeric(col)

    # Create distinct Cohort ID
    df['Cohort_ID'] = df['Buyer'].astype(str).str.replace('.0', '', regex=False) + "-F" + df['Funnel'].astype(str)
    return df

# --- 2. RELATIVE EFFICIENCY & FORECASTING (GoPractice Approach) ---
def perform_benchmarking_analysis(df):
    # Calculate Core Efficiency Metrics
    df['CPC'] = np.where(df['Visits'] > 0, df['Costs'] / df['Visits'], 0)
    df['CVR'] = np.where(df['Visits'] > 0, (df['Regs'] / df['Visits']) * 100, 0)
    df['CPA'] = np.where(df['RFD'] > 0, df['Costs'] / df['RFD'], 0)
    df['ROAS'] = np.where(df['Costs'] > 0, df['In'] / df['Costs'], 0)
    
    # --- PROJECTION ENGINE ---
    # We estimate growth potential. Growth Factor heuristic: 1 + log(Frequency) * Retention
    # High Frequency (repeat deposits) and Low % One timers = High Growth
    df['Retention_Rate'] = (100 - df['% One timers']) / 100
    df['Growth_Factor'] = 1 + np.log1p(df['Frequency Deposit'] - 1).clip(lower=0) * df['Retention_Rate']
    df['Projected_ROAS_6M'] = df['ROAS'] * df['Growth_Factor']
    
    # Calculate Buyer-wide Averages for Benchmarking
    buyer_averages = df.groupby('Buyer').agg({
        'CPA': 'mean',
        'CVR': 'mean',
        'CPC': 'mean'
    }).rename(columns={'CPA': 'Buyer_Avg_CPA', 'CVR': 'Buyer_Avg_CVR', 'CPC': 'Buyer_Avg_CPC'})
    
    df = df.merge(buyer_averages, on='Buyer')

    # Identify "Segment Winners" (Better than average performance)
    df['Efficiency_Score'] = (df['CVR'] / df['Buyer_Avg_CVR']) + (df['Buyer_Avg_CPA'] / df['CPA']).fillna(0)
    
    # Strategic Labeling based on the "Ranking vs Efficiency" theory
    def benchmark_label(row):
        if row['RFD'] < 5:
            return '🔬 SMALL SAMPLE'
        if row['Projected_ROAS_6M'] > 1.3:
            return '💎 SCALE (Future Winner)'
        if row['ROAS'] > 1.2:
            return '✅ PROFITABLE'
        if row['Projected_ROAS_6M'] > 1.0:
            return '🧪 POTENTIAL (High LTV)'
        if row['ROAS'] < 0.7:
             return '🛑 STOP (Burning)'
        return '⚖️ MAINTAIN'

    df['Benchmark_Status'] = df.apply(benchmark_label, axis=1)
    
    return df

# --- 3. REPORTING ENGINE ---
def generate_strategic_report(df):
    print("\n" + "="*80)
    print("🎯 STRATEGIC PERFORMANCE AUDIT (GoPractice Framework)")
    print("="*80)
    
    # Comparison Block: Expected LTV vs Real ROAS
    print("\n>>> 📈 EXPECTED LTV (6M Projection) VS CURRENT ROAS:")
    print("Focusing on cohorts where behavioral signals (Frequency) suggest better long-term value.")
    
    future_winners = df[df['Projected_ROAS_6M'] > df['ROAS']].sort_values(by='Projected_ROAS_6M', ascending=False)
    
    cols = ['Cohort_ID', 'ROAS', 'Projected_ROAS_6M', 'Frequency Deposit', '% One timers', 'Benchmark_Status']
    print(future_winners[cols].head(10).to_markdown(index=False))

    # Identify "Segment Winners" (Better than average performance)
    gems = df[(df['Funnel'] > 2) & (df['ROAS'] > 0.9) & (df['RFD'] >= 5)].sort_values(by='Efficiency_Score', ascending=False)
    
    print("\n>>> 💎 HIDDEN GEMS (Small Volume, High Efficiency):")
    print(gems[['Cohort_ID', 'Costs', 'CPC', 'CVR', 'CPA', 'ROAS', 'Benchmark_Status']].head(5).to_markdown(index=False))

    # Pareto Waste
    big_waste = df[(df['Funnel'] == 1) & (df['ROAS'] < 0.9)].sort_values(by='Costs', ascending=False)
    print("\n>>> ⚠️ BIG BUDGET INEFFICIENCIES (Pareto Waste):")
    if not big_waste.empty:
        print(big_waste[['Cohort_ID', 'Costs', 'CVR', 'CPA', 'ROAS']].head(3).to_markdown(index=False))

# --- MAIN ---
if __name__ == "__main__":
    DATA_PATH = '/Users/yevhen/Cursor/Conv TL/marketing_audit_20260208.csv'
    df = load_and_clean_data(DATA_PATH)
    df_analyzed = perform_benchmarking_analysis(df)
    generate_strategic_report(df_analyzed)
    
    # Save the deep audit
    df_analyzed.to_csv('/Users/yevhen/Cursor/Conv TL/benchmark_audit_results.csv', index=False)
    print("\n✅ Deep audit complete. CSV saved as 'benchmark_audit_results.csv'.")
