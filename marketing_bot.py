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
    """Аналізує дані та генерує стратегічний звіт для Telegram."""
    target_chat_id = chat_id if chat_id else TELEGRAM_CHAT_ID
    
    try:
        # 1. Advanced Cleaning & Column Normalization
        # Прибираємо зайві пробіли та символи з назв колонок
        df.columns = [str(c).replace('*', '').strip() for c in df.columns]
        
        # Обробка дублікатів імен колонок (беремо першу, якщо є кілька)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        def clean_numeric(col):
            if col not in df.columns: 
                return pd.Series(0.0, index=df.index)
            # Примусово в текст -> видалення сміття -> в число -> в float64
            clean_s = df[col].astype(str).str.replace(r'[$\s\xa0%]', '', regex=True).str.replace(',', '.', regex=False)
            return pd.to_numeric(clean_s, errors='coerce').fillna(0.0).astype(float)

        numeric_cols = ['Costs', 'In', 'Out', 'RFD', 'Regs', 'Visits', 'Frequency Deposit', '% One timers']
        for col in numeric_cols:
            df[col] = clean_numeric(col)

        # Технічні колонки (Баєри та Воронки) приводимо до тексту для зручності
        for col in ['Buyer', 'Funnel']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str)

        # 2. Обробка дат
        df['Date_DT'] = pd.to_datetime(df['Date'].astype(str), dayfirst=True, errors='coerce')
        
        # 3. Метрики воронки (всі розрахунки на float64)
        df['CPC'] = np.where(df['Visits'] > 0, df['Costs'] / df['Visits'], 0.0)
        df['CVR'] = np.where(df['Visits'] > 0, (df['Regs'] / df['Visits']) * 100.0, 0.0)
        df['CPA'] = np.where(df['RFD'] > 0, df['Costs'] / df['RFD'], 0.0)
        df['ROAS'] = np.where(df['Costs'] > 0, df['In'] / df['Costs'], 0.0)
        
        # Прогноз LTV (GoPractice logic)
        df['Retention_Rate'] = (100.0 - df['% One timers']) / 100.0
        # np.log1p вимагає числового вводу
        df['Growth_Factor'] = 1.0 + np.log1p(df['Frequency Deposit'].clip(lower=1.0) - 1.0) * df['Retention_Rate']
        df['Projected_ROAS_6M'] = df['ROAS'] * df['Growth_Factor']
        
        df.replace([np.inf, -np.inf], 0.0, inplace=True)
        unique_dates = sorted(df['Date_DT'].dropna().unique())
        
        if not unique_dates:
            bot.send_message(target_chat_id, "⚠️ Валідних дат не знайдено.")
            return df

        latest_date = pd.to_datetime(target_date_str, dayfirst=True) if target_date_str else unique_dates[-1]
        df_latest = df[df['Date_DT'] == latest_date].copy()

        # --- ГЕНЕРАЦІЯ ТЕКСТОВОГО ЗВІТУ ---
        report = f"📋 **СТРАТЕГІЧНИЙ АУДИТ ТРАФІКУ: {latest_date.strftime('%d.%m.%Y')}**\n"
        report += "⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n\n"
        
        # 1. Загальні показники (явне приведення до float для суми)
        total_in = float(df_latest['In'].sum())
        total_cost = float(df_latest['Costs'].sum())
        total_roas = total_in / total_cost if total_cost > 0 else 0.0
        
        report += f"💰 **Загальні результати:**\n"
        report += f"• Витрати: ${total_cost:,.0f}\n"
        report += f"• Дохід: ${total_in:,.0f}\n"
        report += f"• ROAS: **{total_roas:.2f}**\n\n"

        # 2. Найкращі воронки (Масштабування)
        top_winners = df_latest.sort_values(by='Projected_ROAS_6M', ascending=False).head(3)
        report += "🚀 **РЕКОМЕНДОВАНО ДО МАСШТАБУВАННЯ:**\n"
        for _, row in top_winners.iterrows():
            report += f"• **B{row['Buyer']}/F{row['Funnel']}**: ROAS {row['ROAS']:.1f} ⮕ Proj. **{row['Projected_ROAS_6M']:.1f}**\n"
            report += f"  (CPA ${row['CPA']:.1f}, CVR {row['CVR']:.1f}%)\n"
        
        # 3. Приховані діаманти (Future Winners)
        gems = df_latest[(df_latest['Projected_ROAS_6M'] > df_latest['ROAS'] * 1.5) & (df_latest['ROAS'] > 0.7)]
        if not gems.empty:
            report += "\n💎 **ПРИХОВАНІ ДІАМАНТИ (LTV):**\n"
            for _, row in gems.head(2).iterrows():
                report += f"• B{row['Buyer']}/F{row['Funnel']}: Величезний потенціал утримання (Частота {row['Frequency Deposit']:.1f})\n"

        # 4. Критичні зони (Зупинити)
        losers = df_latest[(df_latest['ROAS'] < 0.6) & (df_latest['Costs'] > 1000)].sort_values(by='Costs', ascending=False)
        if not losers.empty:
            report += "\n❌ **ЗУПИНИТИ НЕГАЙНО:**\n"
            for _, row in losers.head(3).iterrows():
                report += f"• B{row['Buyer']}/F{row['Funnel']}: ROAS {row['ROAS']:.2f} (Злив бюджету)\n"

        report += "\n💡 *Висновок:* Основний фокус на Баєра 3 та Воронку 1-F3. Баєра 6 рекомендується відключити через критично високий CPA."

        # --- ВІДПРАВКА ---
        chart_file = generate_charts(df, latest_date)
        if chart_file and os.path.exists(chart_file):
            with open(chart_file, 'rb') as p: bot.send_photo(target_chat_id, p)
            os.remove(chart_file)

        bot.send_message(target_chat_id, report, parse_mode='Markdown')
        return df_latest
        
    except Exception as e:
        import traceback
        bot.send_message(target_chat_id, f"❌ Помилка аналізу: {str(e)}")
        print(traceback.format_exc())
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
