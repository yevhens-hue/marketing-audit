import pandas as pd
import numpy as np
import requests
import os
import sys
import telebot
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8536522580:AAGN2g8NyA5DC2qn65hPMz6rayEj2ISH0gY')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '-5088344832') 
SHEET_ID = '1r-eLYMgcYW1O420YZzIhwr7HItAdDtbU_YdKvw6kCI4'
GID = '1782986040' 
CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}'

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def send_telegram_message(message):
    """Sends a message to the specified Telegram chat."""
    try:
        bot.send_message(TELEGRAM_CHAT_ID, message, parse_mode='Markdown')
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def load_data(chat_id=None):
    """Loads data directly from the Google Sheet."""
    try:
        df = pd.read_csv(CSV_URL, skiprows=1)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].ffill()
        if 'Buyer' in df.columns:
            df['Buyer'] = df['Buyer'].ffill()
        if 'RFD*' in df.columns:
            df.rename(columns={'RFD*': 'RFD'}, inplace=True)
        return df
    except Exception as e:
        error_msg = f"🚨 DATA LOAD ERROR: {str(e)}"
        if chat_id:
            bot.send_message(chat_id, error_msg)
        print(error_msg)
        return None

def generate_charts(df, latest_date):
    """Generates ROAS trend and Cost distribution charts."""
    try:
        # Prepare directory/filename
        filename = "audit_report.png"
        
        # 1. ROAS Trend (Last 7 Days)
        trend_dates = sorted(df['Date_DT'].dropna().unique())
        trend_dates = [d for d in trend_dates if d <= latest_date][-7:]
        
        trend_data = []
        for d in trend_dates:
            day_df = df[df['Date_DT'] == d]
            in_sum = day_df['In'].sum()
            cost_sum = day_df['Costs'].sum()
            roas = in_sum / cost_sum if cost_sum > 0 else 0
            trend_data.append({'Date': d, 'ROAS': roas})
        
        df_trend = pd.DataFrame(trend_data)

        # 2. Cost Distribution (Latest Date)
        df_latest = df[df['Date_DT'] == latest_date]
        buyer_costs = df_latest.groupby('Buyer')['Costs'].sum()

        # Create Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Simple styling
        fig.patch.set_facecolor('#f9f9f9')

        # Plot 1: Trend
        ax1.plot(df_trend['Date'], df_trend['ROAS'], marker='o', color='#2ecc71', linewidth=3, markersize=8)
        ax1.set_title(f'Portfolio ROAS Trend', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('ROAS', fontsize=12)
        ax1.set_xticks(df_trend['Date'])
        ax1.set_xticklabels([d.strftime('%d.%m') for d in df_trend['Date']], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        for x, y in zip(df_trend['Date'], df_trend['ROAS']):
            ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

        # Plot 2: Pie chart for Costs
        if not buyer_costs.empty and buyer_costs.sum() > 0:
            ax2.pie(buyer_costs, labels=[f"B{b}" for b in buyer_costs.index], autopct='%1.1f%%', 
                    startangle=140, pctdistance=0.85, colors=plt.cm.Pastel1.colors)
            # Draw a circle at the center to make it a donut chart
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            ax2.set_title('Cost Distribution', fontsize=14, fontweight='bold', pad=20)
        else:
            ax2.text(0.5, 0.5, 'No Cost Data', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        plt.close()
        return filename
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return None

def analyze_and_report(df, target_date_str=None, chat_id=None):
    """Analyzes the data, adds recommendations, and generates a report."""
    target_chat_id = chat_id if chat_id else TELEGRAM_CHAT_ID
    
    try:
        # 1. Global Fill NaNs with a safe value first, carefully
        # We fill numeric columns with 0 and object columns with empty string or specific default
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype).startswith('string'):
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)

        # 2. Clean numerical columns (Costs, In, etc.)
        cols_to_clean = ['Costs', 'In', 'Out', 'RFD', 'Regs']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[$\s\xa0]', '', regex=True).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 3. Format technical columns (Buyer, Funnel) - ensure they are clean strings
        for col in ['Buyer', 'Funnel']:
            if col in df.columns:
                # Handle cases where it might be 0, "0", or NaN
                temp_numeric = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = temp_numeric.astype(str)

        # 4. Handle Dates
        if 'Date' not in df.columns:
            bot.send_message(target_chat_id, "❌ Error: 'Date' column is missing.")
            return None
            
        df['Date_DT'] = pd.to_datetime(df['Date'].astype(str), dayfirst=True, errors='coerce')
        
        # Calculate Metrics using numpy to avoid division by zero errors before filling
        df['CPA'] = np.where(df['RFD'] != 0, df['Costs'] / df['RFD'], 0)
        df['ROAS'] = np.where(df['Costs'] != 0, df['In'] / df['Costs'], 0)
        
        # Final replacement of any remaining infs (just in case)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        unique_dates = sorted(df['Date_DT'].dropna().unique())
        
        if not unique_dates:
            bot.send_message(target_chat_id, "⚠️ No valid dates found in the sheet.")
            return df

        if target_date_str:
            target_date_str = str(target_date_str).strip()
            try:
                latest_date = pd.to_datetime(target_date_str, dayfirst=True)
                if latest_date not in unique_dates:
                     available_dates = ", ".join([d.strftime('%d.%m.%Y') for d in unique_dates])
                     bot.send_message(target_chat_id, f"⚠️ Date {target_date_str} not found. Available: {available_dates}")
                     return df
            except:
                bot.send_message(target_chat_id, f"⚠️ Invalid date format: {target_date_str}. Use DD.MM.YYYY")
                return df
        else:
            today = pd.Timestamp.now().normalize()
            past_dates = [d for d in unique_dates if d <= today]
            latest_date = past_dates[-1] if past_dates else unique_dates[-1]

        idx = unique_dates.index(latest_date)
        previous_date = unique_dates[idx - 1] if idx > 0 else None
        latest_date_str = latest_date.strftime('%d.%m.%Y')
        
        df_latest = df[df['Date_DT'] == latest_date].copy()
        df_prev = df[df['Date_DT'] == previous_date].copy() if previous_date else df.iloc[:0].copy()
        
        if df_latest.empty:
            bot.send_message(target_chat_id, f"⚠️ No data found for the date {latest_date_str}.")
            return df

        def get_delta_str(current, prev):
            if prev == 0 or pd.isnull(prev): return ""
            delta = ((current - prev) / prev) * 100
            icon = "📈" if delta > 0 else "📉"
            return f" ({icon} {delta:+.1f}%)"

        # --- DECISION LOGIC ---
        recommendations = []
        alerts = []
        for index, row in df_latest.iterrows():
            roas, cost = row['ROAS'], row['Costs']
            buyer, funnel = str(row.get('Buyer', '0')), str(row.get('Funnel', '0'))
            
            rec = "🛡️ MONITOR"
            if roas > 4.0 and cost > 1000:
                rec = "🚀 ROCKET SCALE (ROAS > 4)"
                alerts.append(f"🦄 **UNICORN ALERT:** Buyer {buyer} (Funnel {funnel}) has ROAS {roas:.2f}! Scale!")
            elif roas > 2.0 and cost > 2000:
                rec = "🔥 SCALE AGGRESSIVE"
            elif roas < 0.6 and cost > 2000:
                rec = "❌ STOP IMMEDIATE (Burning Cash)"
                alerts.append(f"🚨 **BUDGET BLEED:** Buyer {buyer} (Funnel {funnel}) ROAS {roas:.2f}. Stop!")
            elif roas > 0.9:
                rec = "📈 MAINTAIN / SCALE"
            recommendations.append(rec)
        
        df_latest['AI_Recommendation'] = recommendations
        
        # --- REPORT TEXT ---
        report = f"📊 **MARKETING AUDIT: {latest_date_str}**\n\n"
        
        # Winning Funnels
        top_winners = df_latest[df_latest['AI_Recommendation'].str.contains('ROCKET|SCALE')].sort_values(by='ROAS', ascending=False).head(5)
        if not top_winners.empty:
            report += "🚀 **TOP OPPORTUNITIES:**\n"
            for _, row in top_winners.iterrows():
                b, f = str(row['Buyer']), str(row['Funnel'])
                prev_row = df_prev[(df_prev['Buyer'] == b) & (df_prev['Funnel'] == f)]
                delta = get_delta_str(row['ROAS'], prev_row['ROAS'].iloc[0]) if not prev_row.empty else ""
                report += f"- B{b}/F{f}: ROAS {row['ROAS']:.2f}{delta}, CPA ${row['CPA']:.1f}\n"
            
        # Losing Funnels
        top_losers = df_latest[df_latest['AI_Recommendation'].str.contains('STOP')].sort_values(by='Costs', ascending=False).head(5)
        if not top_losers.empty:
            report += "\n❌ **CRITICAL CUTS:**\n"
            for _, row in top_losers.iterrows():
                b, f = str(row['Buyer']), str(row['Funnel'])
                prev_row = df_prev[(df_prev['Buyer'] == b) & (df_prev['Funnel'] == f)]
                delta = get_delta_str(row['ROAS'], prev_row['ROAS'].iloc[0]) if not prev_row.empty else ""
                report += f"- B{b}/F{f}: ROAS {row['ROAS']:.2f}{delta}, Cost ${row['Costs']:.0f}\n"
            
        if previous_date:
            t_in_now, t_in_prev = df_latest['In'].sum(), df_prev['In'].sum()
            costs_now = df_latest['Costs'].sum()
            costs_prev = df_prev['Costs'].sum()
            roas_now = t_in_now / costs_now if costs_now > 0 else 0
            roas_prev = t_in_prev / costs_prev if costs_prev > 0 else 0
            
            report += f"\n📈 **DoD Summary:**\n"
            report += f"- Revenue: ${t_in_now:,.0f}{get_delta_str(t_in_now, t_in_prev)}\n"
            report += f"- Portf. ROAS: {roas_now:.2f}{get_delta_str(roas_now, roas_prev)}\n"

        if alerts:
            report += "\n⚠️ **ALERTS:**\n" + "\n".join(alerts)
            
        # --- GENERATE AND SEND CHARTS ---
        chart_file = generate_charts(df, latest_date)
        if chart_file and os.path.exists(chart_file):
            try:
                with open(chart_file, 'rb') as photo:
                    bot.send_photo(target_chat_id, photo)
                os.remove(chart_file) # Clean up
            except Exception as chart_err:
                print(f"Failed to send chart: {chart_err}")

        bot.send_message(target_chat_id, report, parse_mode='Markdown')
        return df_latest
        
    except Exception as e:
        bot.send_message(target_chat_id, f"❌ Analysis Crash: {str(e)}")
        print(f"Crash details: {e}")
        return None

# --- TELEGRAM BOT HANDLERS ---
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = "👋 **Marketing Audit Bot + Charts**\n\nCommands:\n/report - Full audit with charts\n/report DD.MM.YYYY - Audit for specific date\n/status - Check connection"
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['report'])
def handle_report_command(message):
    chat_id = message.chat.id
    try:
        args = message.text.split()
        target_date = args[1] if len(args) > 1 else None
        
        waiting_msg = bot.reply_to(message, "⏳ Connecting to Google Sheets...")
        
        df = load_data(chat_id=chat_id)
        if df is None: return
            
        bot.edit_message_text("📊 Aggregating and Analyzing metrics...", chat_id, waiting_msg.message_id)
        
        analyze_and_report(df, target_date_str=target_date, chat_id=chat_id)
        
        bot.delete_message(chat_id, waiting_msg.message_id)
        
    except Exception as e:
        bot.send_message(chat_id, f"❌ Command Error: {str(e)}")

@bot.message_handler(commands=['status'])
def handle_status(message):
    bot.reply_to(message, "✅ System is Online and connected to Google Sheets.")

# --- MAIN EXECUTION ---
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bot":
        print("Starting Telegram Bot Polling Mode...")
        bot.infinity_polling()
    else:
        target_date = sys.argv[1] if len(sys.argv) > 1 else None
        print(f"Running One-off Audit (Target: {target_date if target_date else 'LATEST'})...")
        df = load_data()
        if df is not None:
            analyze_and_report(df, target_date_str=target_date)
            print("Done! Report sent to Telegram.")
