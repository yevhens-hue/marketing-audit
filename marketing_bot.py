import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# --- CONFIGURATION ---
# It's better to use Environment Variables for security
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8536522580:AAGN2g8NyA5DC2qn65hPMz6rayEj2ISH0gY')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '-5088344832') 
# Google Sheet URL (Publicly accessible or setup with service account)
SHEET_ID = '1r-eLYMgcYW1O420YZzIhwr7HItAdDtbU_YdKvw6kCI4'
GID = '1782986040' 
CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}'

def send_telegram_message(message):
    """Sends a message to the specified Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def load_data():
    """Loads data directly from the Google Sheet."""
    try:
        # Added skiprows=1 to skip the description row
        df = pd.read_csv(CSV_URL, skiprows=1)
        
        # Forward fill Date and Buyer columns to handle merged cells in Google Sheets
        if 'Date' in df.columns:
            df['Date'] = df['Date'].ffill()
        if 'Buyer' in df.columns:
            df['Buyer'] = df['Buyer'].ffill()
            
        # Rename "RFD*" to "RFD" to match our logic
        if 'RFD*' in df.columns:
            df.rename(columns={'RFD*': 'RFD'}, inplace=True)
            
        return df
    except Exception as e:
        send_telegram_message(f"🚨 CRITICAL ERROR: Failed to load data from Google Sheet.\nError: {str(e)}")
        return None

def analyze_and_report(df):
    """Analyzes the data, adds recommendations, and generates a report."""
    
    # Clean and convert numeric columns
    cols_to_clean = ['Costs', 'In', 'Out', 'RFD', 'Regs']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[$\s\xa0]', '', regex=True).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate Metrics for the entire dataset
    df['CPA'] = df['Costs'] / df['RFD']
    df['ROAS'] = df['In'] / df['Costs']
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # --- FILTER BY DATE ---
    df['Date_DT'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Get all unique dates sorted
    unique_dates = sorted(df['Date_DT'].dropna().unique())
    if not unique_dates:
        send_telegram_message("⚠️ No valid dates found in the sheet.")
        return df

    latest_date = unique_dates[-1]
    previous_date = unique_dates[-2] if len(unique_dates) > 1 else None
    
    latest_date_str = latest_date.strftime('%d/%m/%Y')
    
    # Filter dataframes
    df_latest = df[df['Date_DT'] == latest_date].copy()
    # Initialize df_prev with columns even if empty
    df_prev = df[df['Date_DT'] == previous_date].copy() if previous_date else df.iloc[:0].copy()
    
    if df_latest.empty:
        send_telegram_message(f"⚠️ No data found for the latest date {latest_date_str}.")
        return df

    def get_delta_str(current, prev):
        if prev == 0 or pd.isnull(prev):
            return ""
        delta = ((current - prev) / prev) * 100
        icon = "📈" if delta > 0 else "📉"
        return f" ({icon} {delta:+.1f}%)"

    # --- DECISION LOGIC (on latest data) ---
    recommendations = []
    alerts = []
    
    for index, row in df_latest.iterrows():
        roas = row['ROAS']
        cost = row['Costs']
        buyer = row.get('Buyer', 'Unknown')
        funnel = row.get('Funnel', 'Unknown')
        
        rec = "🛡️ MONITOR"
        if roas > 4.0 and cost > 1000:
            rec = "🚀 ROCKET SCALE (ROAS > 4)"
            alerts.append(f"🦄 **UNICORN ALERT:** Buyer {buyer} (Funnel {funnel}) has ROAS {roas:.2f}! Scale immediately!")
        elif roas > 2.0 and cost > 2000:
            rec = "🔥 SCALE AGGRESSIVE"
        elif roas < 0.6 and cost > 2000:
            rec = "❌ STOP IMMEDIATE (Burning Cash)"
            alerts.append(f"🚨 **BUDGET BLEED:** Buyer {buyer} (Funnel {funnel}) is burning cash (ROAS {roas:.2f}). Stop now.")
        elif roas > 0.9:
            rec = "📈 MAINTAIN / SCALE"
            
        recommendations.append(rec)
    
    df_latest['AI_Recommendation'] = recommendations
    
    # --- GENERATE REPORT TEXT ---
    report = f"📊 **DAILY MARKETING AUDIT: {latest_date_str}**\n"
    report += f"*(Analysis based on {len(df_latest)} entries)*\n\n"
    
    # Top Winners with comparison
    top_winners = df_latest[df_latest['AI_Recommendation'].str.contains('ROCKET|SCALE')].sort_values(by='ROAS', ascending=False).head(5)
    if not top_winners.empty:
        report += "🚀 **TOP OPPORTUNITIES:**\n"
        for _, row in top_winners.iterrows():
            buyer, funnel = row.get('Buyer'), row.get('Funnel')
            # Try to find previous performance for this specific buyer/funnel
            prev_row = df_prev[(df_prev['Buyer'] == buyer) & (df_prev['Funnel'] == funnel)]
            roas_delta = get_delta_str(row['ROAS'], prev_row['ROAS'].values[0]) if not prev_row.empty else ""
            
            report += f"- Buyer {buyer}/F{funnel}: ROAS {row['ROAS']:.2f}{roas_delta}, CPA ${row['CPA']:.0f} ({row['AI_Recommendation']})\n"
        
    # Top Losers with comparison
    top_losers = df_latest[df_latest['AI_Recommendation'].str.contains('STOP')].sort_values(by='Costs', ascending=False).head(5)
    if not top_losers.empty:
        report += "\n❌ **CRITICAL CUTS:**\n"
        for _, row in top_losers.iterrows():
            buyer, funnel = row.get('Buyer'), row.get('Funnel')
            prev_row = df_prev[(df_prev['Buyer'] == buyer) & (df_prev['Funnel'] == funnel)]
            roas_delta = get_delta_str(row['ROAS'], prev_row['ROAS'].values[0]) if not prev_row.empty else ""
            
            report += f"- Buyer {buyer}/F{funnel}: ROAS {row['ROAS']:.2f}{roas_delta}, Cost ${row['Costs']:.0f} ({row['AI_Recommendation']})\n"
        
    # DoD Summary
    if previous_date:
        total_in_now = df_latest['In'].sum()
        total_in_prev = df_prev['In'].sum()
        in_delta = get_delta_str(total_in_now, total_in_prev)
        
        avg_roas_now = df_latest['In'].sum() / df_latest['Costs'].sum() if df_latest['Costs'].sum() > 0 else 0
        avg_roas_prev = df_prev['In'].sum() / df_prev['Costs'].sum() if df_prev['Costs'].sum() > 0 else 0
        roas_delta_total = get_delta_str(avg_roas_now, avg_roas_prev)
        
        report += f"\n📈 **DoD Summary:**\n"
        report += f"- Total Revenue: ${total_in_now:,.0f}{in_delta}\n"
        report += f"- Portfolio ROAS: {avg_roas_now:.2f}{roas_delta_total}\n"

    # Append Alerts
    if alerts:
        report += "\n⚠️ **CRITICAL ALERTS:**\n"
        report += "\n".join(alerts)
        
    send_telegram_message(report)
    return df_latest

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    if df is not None:
        print("Analyzing...")
        result_df = analyze_and_report(df)
        
        # Save locally with recommendations
        output_filename = f"marketing_audit_{datetime.now().strftime('%Y%m%d')}.csv"
        result_df.to_csv(output_filename, index=False)
        print(f"Done! Report saved to {output_filename} and sent to Telegram.")
