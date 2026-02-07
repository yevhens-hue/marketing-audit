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
    # Handle currency like "$18 094" (remove spaces, $, replace comma with dot if needed)
    cols_to_clean = ['Costs', 'In', 'Out', 'RFD', 'Regs']
    for col in cols_to_clean:
        if col in df.columns:
            # Convert to string, replace (non-breaking) spaces and $, replace comma with dot
            df[col] = df[col].astype(str).str.replace(r'[$\s\xa0]', '', regex=True).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate Metrics
    df['CPA'] = df['Costs'] / df['RFD']
    df['ROAS'] = df['In'] / df['Costs']
    
    # Handle division by zero
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- DECISION LOGIC ---
    recommendations = []
    alerts = []
    
    for index, row in df.iterrows():
        roas = row['ROAS']
        cost = row['Costs']
        buyer = row.get('Buyer', 'Unknown')
        funnel = row.get('Funnel', 'Unknown')
        
        rec = "🛡️ MONITOR"
        
        # Logic Tree
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
    
    df['AI_Recommendation'] = recommendations
    
    # --- GENERATE REPORT TEXT ---
    
    top_winners = df[df['AI_Recommendation'].str.contains('ROCKET|SCALE')].sort_values(by='ROAS', ascending=False).head(5)
    top_losers = df[df['AI_Recommendation'].str.contains('STOP')].sort_values(by='Costs', ascending=False).head(5)
    
    report = f"📊 **DAILY MARKETING AUDIT: {datetime.now().strftime('%Y-%m-%d')}**\n\n"
    
    report += "🚀 **TOP OPPORTUNITIES:**\n"
    for _, row in top_winners.iterrows():
        report += f"- Buyer {row.get('Buyer')}/F{row.get('Funnel')}: ROAS {row['ROAS']:.2f}, CPA ${row['CPA']:.0f} ({row['AI_Recommendation']})\n"
        
    report += "\n❌ **CRITICAL CUTS:**\n"
    for _, row in top_losers.iterrows():
        report += f"- Buyer {row.get('Buyer')}/F{row.get('Funnel')}: ROAS {row['ROAS']:.2f}, Cost ${row['Costs']:.0f} ({row['AI_Recommendation']})\n"
        
    # Append Alerts to the same report message if any
    if alerts:
        report += "\n⚠️ **CRITICAL ALERTS:**\n"
        report += "\n".join(alerts)
        
    # Send everything as ONE message
    send_telegram_message(report)
            
    return df

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
