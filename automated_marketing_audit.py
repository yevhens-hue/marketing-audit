import pandas as pd
import numpy as np

# --- DATASET: Emulating the input data from your task ---
# In production, you would use: df = pd.read_csv('data.csv')

buyers_data = {
    'Buyer': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 14, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22],
    'Funnel': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 6, 8, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 6, 1, 2, 3, 1, 2, 3],
    'Costs': [18094, 12079, 11414, 2709, 2139, 1390, 1155, 451, 310, 48063, 190, 15757, 6736, 6679, 6166, 3383, 1184, 515, 281, 12861, 8681, 2881, 2097, 1592, 977, 16235, 2700, 2293, 2060, 1563, 1262, 770, 230, 18188, 306, 11678, 4147, 5949, 5243, 2114, 1133, 562, 7521, 7190, 3105, 2254, 1738, 1188, 1005, 924, 908, 360, 312, 2721, 2313, 1983, 1856, 1854, 1658, 735, 4324, 2277, 2247, 2016, 212, 124, 7756, 3112, 11335, 10703, 250, 6114, 2361, 1472, 8620, 1050, 313, 4833, 3642, 431, 8180, 541, 6913, 459, 189, 6257, 1050, 735, 4990, 1827, 153],
    'RFD': [98, 88, 96, 12, 15, 11, 5, 2, 2, 253, 0, 153, 47, 62, 59, 29, 3, 2, 4, 89, 55, 18, 21, 8, 8, 112, 17, 10, 15, 11, 5, 4, 1, 84, 2, 67, 14, 31, 14, 14, 7, 2, 25, 43, 9, 9, 8, 2, 6, 2, 3, 1, 1, 12, 13, 11, 7, 5, 7, 3, 29, 31, 8, 6, 1, 1, 38, 21, 68, 65, 1, 52, 12, 8, 42, 2, 2, 24, 27, 1, 36, 1, 37, 3, 1, 28, 2, 2, 12, 3, 0],
    'In': [14265, 11755, 14580, 2175, 3710, 1465, 371, 781, 456, 42711, 0, 16930, 17225, 15133, 15052, 21433, 216, 98, 152, 12180, 5828, 2334, 1181, 1570, 221, 34503, 7240, 835, 2920, 555, 208, 118, 215, 9839, 107, 6135, 11309, 2467, 2617, 531, 417, 761, 2437, 5230, 633, 2124, 5428, 463, 208, 271, 396, 7, 1193, 1277, 5991, 1791, 775, 232, 3437, 170, 6391, 9998, 750, 215, 34, 21, 7246, 2629, 8144, 14585, 35, 4984, 406, 532, 3707, 4546, 1649, 2944, 3526, 145, 6745, 237, 2432, 444, 21, 1352, 166, 133, 1267, 159, 0]
}

df_buyers = pd.DataFrame(buyers_data)

geo_data = {
    'Geo': ['Geo 1', 'Geo 1', 'Geo 2', 'Geo 2', 'Geo 3', 'Geo 3', 'Geo 4', 'Geo 5'],
    'Product': [1, 2, 1, 2, 1, 2, 1, 2],
    'Costs': [453207, 252522, 342367, 268776, 171159, 135302, 199500, 62080],
    'Regs': [7944, 5855, 5110, 3514, 3121, 2574, 7030, 829],
    'RFD': [2579, 1746, 1539, 1119, 676, 533, 932, 254],
    'In': [432767, 238355, 260243, 158629, 144017, 116448, 126794, 63040]
}

df_geo = pd.DataFrame(geo_data)

# --- AUTOMATED ANALYSIS ENGINE ---

def analyze_buyers(df):
    # Calculate Core Metrics
    df['CPA'] = df['Costs'] / df['RFD']
    df['ROAS'] = df['In'] / df['Costs']
    
    # Avoid div/0
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Automated Decision Logic
    strategies = []
    for _, row in df.iterrows():
        roas = row['ROAS']
        cost = row['Costs']
        cpa = row['CPA']
        
        # Skip small experiments
        if cost < 1000 and roas < 3:
             rec = "🛡️ MONITOR"
        elif roas > 4.0 and cost > 1000:
            rec = "🚀 ROCKET SCALE (ROAS > 4)"
        elif roas > 2.0 and cost > 2000:
            rec = "🔥 SCALE AGGRESSIVE (ROAS > 2)"
        elif roas > 0.8 and cost > 10000:
            rec = "⚠️ OPTIMIZE (High Vol, Low ROAS)"
        elif roas < 0.6 and cost > 2000:
            rec = "❌ STOP IMMEDIATE (Burning Cash)"
        elif cpa > 250 and cost > 1000:
             rec = "❌ STOP (CPA too high)"
        elif roas > 0.9:
            rec = "📈 MAINTAIN / SCALE"
        else:
            rec = "🛡️ MONITOR"
        strategies.append(rec)
    
    df['AI_Recommendation'] = strategies
    return df

def analyze_geo(df):
    df['ROAS'] = df['In'] / df['Costs']
    df['CPA'] = df['Costs'] / df['RFD']
    df['Reg2Dep'] = (df['RFD'] / df['Regs']) * 100
    
    recs = []
    for _, row in df.iterrows():
        if row['ROAS'] > 1.0:
            recs.append("💎 HIDDEN GEM (Double Budget)")
        elif row['Reg2Dep'] < 15.0:
            recs.append("🔧 FIX FUNNEL (Low CR)")
        elif row['ROAS'] < 0.6:
            recs.append("✂️ CUT BUDGET (Unprofitable)")
        elif row['ROAS'] > 0.9:
            recs.append("🛡️ SCALE / MAINTAIN")
        else:
            recs.append("🛡️ MONITOR")
    df['Strategy'] = recs
    return df

# Run Analysis
final_buyers = analyze_buyers(df_buyers)
final_geo = analyze_geo(df_geo)

# Filter for Report
top_opportunities = final_buyers[final_buyers['AI_Recommendation'].str.contains('ROCKET|SCALE')].sort_values(by='ROAS', ascending=False)
critical_cuts = final_buyers[final_buyers['AI_Recommendation'].str.contains('STOP')].sort_values(by='Costs', ascending=False)
geo_report = final_geo.sort_values(by='ROAS', ascending=False)

# --- PRINT REPORT ---
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\n" + "="*80)
print("🤖 AUTOMATED MARKETING PERFORMANCE AUDIT (AUTO-GENERATED)")
print("="*80)

print("\n>>> 🚀 TOP SCALING OPPORTUNITIES (Money Printing Machines):")
print(top_opportunities[['Buyer', 'Funnel', 'Costs', 'RFD', 'CPA', 'ROAS', 'AI_Recommendation']].head(5).to_string(index=False))

print("\n>>> ❌ BUDGET BLEEDING (Where to cut immediately):")
print(critical_cuts[['Buyer', 'Funnel', 'Costs', 'RFD', 'CPA', 'ROAS', 'AI_Recommendation']].head(5).to_string(index=False))

print("\n>>> 🌍 GEO STRATEGY MATRIX:")
print(geo_report[['Geo', 'Product', 'Costs', 'ROAS', 'Reg2Dep', 'Strategy']].to_string(index=False))

best_roas = final_buyers['ROAS'].max()
worst_buyer_id = critical_cuts.iloc[0]['Buyer'] if not critical_cuts.empty else 0
worst_buyer_roas = critical_cuts.iloc[0]['ROAS'] if not critical_cuts.empty else 0

print("\n" + "-"*80)
print(f"💡 EXECUTIVE SUMMARY:")
print(f"1. **UNICORN ALERT:** Found a funnel with ROAS {best_roas:.2f}. Immediate scaling recommended.")
print(f"2. **CRITICAL LEAK:** Buyer {int(worst_buyer_id)} is burning cash with ROAS {worst_buyer_roas:.2f}. Stop immediately.")
print("3. **GEO PIVOT:** Geo 5 Product 2 is the only profitable Geo (ROAS > 1.0). Reallocate budget there.")
print("="*80)
