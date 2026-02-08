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
GID_TAB1 = '1782986040' 
GID_TAB2 = '407212296'
CSV_URL_TAB1 = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_TAB1}'
CSV_URL_TAB2 = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_TAB2}'

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def send_telegram_message(message):
    """Sends a message to the specified Telegram chat."""
    try:
        bot.send_message(TELEGRAM_CHAT_ID, message, parse_mode='Markdown')
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def load_data(chat_id=None, tab=1):
    """Loads data from the specified tab of the Google Sheet."""
    url = CSV_URL_TAB1 if tab == 1 else CSV_URL_TAB2
    try:
        df = pd.read_csv(url, skiprows=1 if tab == 1 else 0)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].ffill()
        if 'Buyer' in df.columns:
            df['Buyer'] = df['Buyer'].ffill()
        if tab == 1 and 'RFD*' in df.columns:
            df.rename(columns={'RFD*': 'RFD'}, inplace=True)
        return df
    except Exception as e:
        error_msg = f"🚨 DATA LOAD ERROR (TAB {tab}): {str(e)}"
        if chat_id:
            bot.send_message(chat_id, error_msg)
        print(error_msg)
        return None

def generate_charts(df, latest_date):
    """Generates a comprehensive Dashboard (Charts + Table)."""
    try:
        filename = "dashboard_report.png"
        
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

        # 2. Buyer Performance (Latest Date)
        df_latest = df[df['Date_DT'] == latest_date]
        buyer_perf = df_latest.groupby('Buyer').agg({'Costs': 'sum', 'In': 'sum'}).reset_index()

        # 3. Summary Table Data
        table_data = []
        # Top rows by Revenue
        top_rows = df_latest.sort_values(by='In', ascending=False).head(8)
        for _, row in top_rows.iterrows():
            table_data.append([
                f"B{row.get('Buyer')}/F{row.get('Funnel')}",
                f"${row['Costs']:,.0f}",
                int(row['RFD']),
                f"{row['ROAS']:.2f}",
                f"{row['CR_Reg2Dep']:.1f}%"
            ])

        # Create Dashboard Layout
        # 2 rows: top row has 2 charts, bottom row has the table
        fig = plt.figure(figsize=(16, 12))
        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)
        fig.patch.set_facecolor('#f4f7f6')

        # Ax1: ROAS Trend
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.plot(df_trend['Date'], df_trend['ROAS'], marker='o', color='#3498db', linewidth=3, markersize=10, label='Portfolio ROAS')
        ax1.set_title('7-Day ROAS Trend', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('ROAS', fontsize=12)
        ax1.set_xticks(df_trend['Date'])
        ax1.set_xticklabels([d.strftime('%d.%m') for d in df_trend['Date']], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax1.legend()
        for x, y in zip(df_trend['Date'], df_trend['ROAS']):
            ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#2c3e50')

        # Ax2: Buyer Costs vs Revenue
        ax2 = fig.add_subplot(grid[0, 1])
        x_indices = np.arange(len(buyer_perf))
        width = 0.35
        ax2.bar(x_indices - width/2, buyer_perf['Costs'], width, label='Costs', color='#e74c3c', alpha=0.8)
        ax2.bar(x_indices + width/2, buyer_perf['In'], width, label='Revenue', color='#27ae60', alpha=0.8)
        ax2.set_title('Costs vs Revenue by Buyer', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels([f"B{b}" for b in buyer_perf['Buyer']])
        ax2.set_ylabel('Amount ($)')
        ax2.legend()
        ax2.grid(True, axis='y', linestyle='--', alpha=0.2)

        # Ax3: Summary Table
        ax3 = fig.add_subplot(grid[1, :])
        ax3.axis('off')
        col_labels = ['Buyer/Funnel', 'Costs', 'RFD', 'ROAS', 'Reg2Dep CR']
        if table_data:
            the_table = ax3.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(13)
            the_table.scale(1, 2.8)
            
            # Color headers
            for (row_idx, col_idx), cell in the_table.get_celld().items():
                if row_idx == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2c3e50')
                else:
                    if col_idx == 3: # ROAS column
                        try:
                            val = float(table_data[row_idx-1][3])
                            if val >= 2.0: cell.get_text().set_color('#27ae60')
                            elif val < 1.0: cell.get_text().set_color('#e74c3c')
                        except: pass
        
        ax3.set_title(f'Performance Summary Table: {latest_date.strftime("%d.%m.%Y")}', fontsize=18, fontweight='bold', y=0.95)

        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return None

def generate_tab2_report(df_tab2, latest_date):
    """Generates charts and summary for Tab 2 (Aggregated Data)."""
    try:
        filename = "tab2_report.png"
        
        # Clean numeric data for Tab 2
        for col in ['Costs', 'In', 'Out', 'RFD', 'Regs']:
            if col in df_tab2.columns:
                df_tab2[col] = df_tab2[col].astype(str).str.replace(r'[$\s\xa0]', '', regex=True).str.replace(',', '.')
                df_tab2[col] = pd.to_numeric(df_tab2[col], errors='coerce').fillna(0)
        
        df_tab2['Date_DT'] = pd.to_datetime(df_tab2['Date'].astype(str), dayfirst=True, errors='coerce')
        
        # Filter for latest date
        df_latest = df_tab2[df_tab2['Date_DT'] == latest_date]
        if df_latest.empty:
            return None, "No specific data for this date in Tab 2"

        # Pivot data for a table-like summary
        summary_group = 'Buyer' if 'Buyer' in df_tab2.columns else df_tab2.columns[1]
        summary = df_latest.groupby(summary_group).agg({
            'Costs': 'sum',
            'In': 'sum',
            'RFD': 'sum'
        })
        summary['ROAS'] = np.where(summary['Costs'] > 0, summary['In'] / summary['Costs'], 0)
        summary.fillna(0, inplace=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#ffffff')
        
        x = np.arange(len(summary))
        width = 0.35
        ax.bar(x - width/2, summary['Costs'], width, label='Costs', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, summary['In'], width, label='In (Revenue)', color='#2ecc71', alpha=0.8)
        
        ax.set_title(f'Tab 2 Performance: {latest_date.strftime("%d.%m.%Y")}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index)
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=120)
        plt.close()

        # Text Summary
        text_report = "📖 **TAB 2 (Aggregated) DETAILS:**\n"
        for idx, row in summary.iterrows():
            text_report += f"- {idx}: Cost ${row['Costs']:,.0f} | In ${row['In']:,.0f} | ROAS {row['ROAS']:.2f}\n"
            
        return filename, text_report
    except Exception as e:
        print(f"Tab 2 report failed: {e}")
        return None, None

def analyze_and_report(df, target_date_str=None, chat_id=None, df_tab2=None):
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
        df['Profit'] = df['In'] - df['Costs']
        df['CR_Reg2Dep'] = np.where(df['Regs'] != 0, (df['RFD'] / df['Regs']) * 100, 0)
        
        # Final replacement of any remaining infs
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
            roas, cost, cr = row['ROAS'], row['Costs'], row['CR_Reg2Dep']
            buyer, funnel = str(row.get('Buyer', '0')), str(row.get('Funnel', '0'))
            
            rec = "🛡️ MONITOR"
            if roas > 4.0 and cost > 1000:
                rec = "🚀 ROCKET SCALE (ROAS > 4)"
                alerts.append(f"🦄 **UNICORN ALERT:** Buyer {buyer}/F{funnel} (ROAS {roas:.2f})")
            elif roas < 0.6 and cost > 1000:
                rec = "❌ STOP IMMEDIATE"
                alerts.append(f"🚨 **BUDGET BLEED:** Buyer {buyer}/F{funnel} (ROAS {roas:.2f})")
            
            # CR-based Alert
            if cr < 5.0 and row['Regs'] > 20:
                alerts.append(f"🗑️ **TRASH TRAFFIC:** Buyer {buyer}/F{funnel} CR {cr:.1f}% (Low Quality)")

            recommendations.append(rec)
        
        df_latest['AI_Recommendation'] = recommendations
        
        # --- CALC OPTIMIZATION FORECAST ---
        df_optimized = df_latest[~df_latest['AI_Recommendation'].str.contains('STOP')].copy()
        opt_roas = df_optimized['In'].sum() / df_optimized['Costs'].sum() if df_optimized['Costs'].sum() > 0 else 0
        current_roas = df_latest['In'].sum() / df_latest['Costs'].sum() if df_latest['Costs'].sum() > 0 else 0
        
        # --- REPORT TEXT ---
        report = f"📊 **MARKETING AUDIT: {latest_date_str}**\n\n"
        
        # Portfolio Overview
        total_in = df_latest['In'].sum()
        total_cost = df_latest['Costs'].sum()
        total_profit = df_latest['Profit'].sum()
        avg_cr = (df_latest['RFD'].sum() / df_latest['Regs'].sum() * 100) if df_latest['Regs'].sum() > 0 else 0
        
        report += f"💰 **Main Results:**\n"
        report += f"- Profit: ${total_profit:,.0f} (ROI {(((total_in-total_cost)/total_cost*100) if total_cost > 0 else 0):.1f}%)\n"
        report += f"- Avg CR (Reg2Dep): {avg_cr:.1f}%\n\n"

        # Optimization Insight
        if opt_roas > current_roas:
            report += f"💡 **AI OPTIMIZER:**\nStopping 'STOP' campaigns would boost ROAS from **{current_roas:.2f}** up to **{opt_roas:.2f}**! 🚀\n\n"

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

        # --- TAB 2 REPORTING ---
        if df_tab2 is not None:
            tab2_file, tab2_text = generate_tab2_report(df_tab2, latest_date)
            if tab2_file and os.path.exists(tab2_file):
                try:
                    with open(tab2_file, 'rb') as photo:
                        bot.send_photo(target_chat_id, photo)
                    os.remove(tab2_file)
                except: pass
            if tab2_text:
                bot.send_message(target_chat_id, tab2_text, parse_mode='Markdown')

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
        
        df = load_data(chat_id=chat_id, tab=1)
        df_tab2 = load_data(chat_id=chat_id, tab=2)
        
        if df is None: return
            
        bot.edit_message_text("📊 Aggregating and Analyzing metrics...", chat_id, waiting_msg.message_id)
        
        analyze_and_report(df, target_date_str=target_date, chat_id=chat_id, df_tab2=df_tab2)
        
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
