import datetime
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import locale
from dateutil.relativedelta import relativedelta
from tkinter import Tk, Text, Scrollbar, RIGHT, Y, END
import threading
import time
import math
import warnings

warnings.simplefilter('ignore')

locale.setlocale(locale.LC_ALL, 'en_US')

base_url = r'https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/slices/CFTC_SLICE_RATES_{:%Y_%m_%d}_{}.zip'

mpc = {("2024-06-20", "2024-08-01"), ("2024-08-01", "2024-09-19"), ("2024-09-19", "2024-11-07"), ("2024-11-07", "2024-12-19"),("2024-12-19", "2025-02-06"), ("2025-02-06", "2025-03-20"),("2025-03-20", "2025-05-08"),("2025-05-08", "2025-06-19"),("2025-06-19", "2025-08-07")}
fomc = {("2024-06-12", "2024-07-31"), ("2024-07-31", "2024-09-18"), ("2024-09-18", "2024-11-07"), ("2024-11-07", "2024-12-18")}
ecb = {("2024-06-12", "2024-07-24"), ("2024-07-24", "2024-09-18"), ("2024-09-18", "2024-10-23"), ("2024-10-23", "2024-12-18"), ("2024-12-18","2025-02-05"), ("2025-02-05","2025-03-12"), ("2025-03-12","2025-04-23"), ("2025-04-23","2025-06-11"),("2025-06-11","2025-07-30")}
cad = {("2024-07-25", "2024-09-05"), ("2024-09-05", "2024-10-24"), ("2024-10-24", "2024-12-12")}
aud = {("2024-08-07", "2024-09-25"), ("2024-09-25", "2024-11-06"), ("2024-11-06", "2024-12-11"), ("2024-12-11", "2025-02-19"), ("2025-02-19","2025-04-02"), ("2025-04-02","2025-05-21"), ("2025-05-21","2025-07-09"),("2025-07-09","2025-08-13")}
nzd = {("2024-07-11", "2024-08-15"), ("2024-08-15", "2024-10-10"), ("2024-10-10", "2024-11-28"), ("2024-11-28", "2025-02-20"),("2025-02-20", "2025-04-10"),("2025-04-10","2025-05-29"),("2025-05-29","2025-07-10")}
jpy = {("2024-08-01", "2024-09-20"), ("2024-09-24", "2024-10-31"), ("2024-11-01", "2024-12-19")}

def get_dv01(notional, rate, effective, maturity):
    days_to_maturity = (maturity - effective).days
    years_to_maturity = days_to_maturity / 360.0
    dv01_leg_1 = (notional * rate * years_to_maturity) / 10000
    dv01_leg_1 = (dv01_leg_1 * 2) / 100
    return dv01_leg_1

def get_dtcc_file(ref_date: datetime.date, counter: int) -> bytes:
    final_url = base_url.format(ref_date, counter)
    resp = requests.get(final_url)
    if resp.status_code != 200:
        return None
    return resp.content

def create_text(effective_date, event_type, rate, currency, dv01, timestamp):
    rate_percent = round(rate * 100, 7)  # Convert rate to percentage and round to 3sf
    currency_symbol = "£" if currency == "GBP" else ("$" if currency == "USD" else "€")
    dv01_text = f"{currency_symbol}{dv01:.3f}k"
    timestamp_dt = pd.to_datetime(timestamp)
    timestamp_dt_plus_one_hour = timestamp_dt + pd.Timedelta(hours=1)
    timestamp_text = f"Time Stamp: {timestamp_dt_plus_one_hour.strftime('%Y-%m-%d %H:%M:%S')[-9:-1]}"
    
    text = f"{effective_date.strftime('%B')} {event_type} trades at {rate_percent}%\n"
    text += f"with DV01 {dv01_text}\n"
    text += timestamp_text
    
    return text

def show_notification(text):
    root = Tk()
    root.overrideredirect(True)
    
    # Set the size of the window
    window_width = 400
    window_height = 200
    
    # Set the window position to bottom left
    position_right = 10
    position_down = root.winfo_screenheight() - window_height - 50

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

    # Create a Text widget
    text_widget = Text(root, height=10, width=50, wrap='word', font=("Helvetica", 14))
    text_widget.insert(END, text)
    text_widget.configure(state='normal')  # Make the text widget editable (for copying)
    text_widget.pack(padx=10, pady=10)

    # Add a scrollbar
    scrollbar = Scrollbar(root, command=text_widget.yview)
    text_widget['yscrollcommand'] = scrollbar.set
    scrollbar.pack(side=RIGHT, fill=Y)

    # Close the window after 30 seconds
    root.after(30000, root.destroy)
    
    root.mainloop()

def send_notification(event_type, effective_date, rate, currency, dv01, timestamp):
    if dv01 > 10:  # Only show notification if DV01 is greater than 10
        text = create_text(effective_date, event_type, rate, currency, dv01, timestamp)
        show_notification(text)

ref_date = datetime.date.today() #- datetime.timedelta(days=3)
counter = 3110

print(f"Processing counter: {counter}")
zip_content = get_dtcc_file(ref_date, counter)

with ZipFile(BytesIO(zip_content)) as z:
    file_list = z.namelist()
    
    with z.open(file_list[0]) as f:
        df = pd.read_csv(f)
        df.to_csv(f'attempt2_counter_{counter}.csv')

        selected_columns = [
            "Dissemination Identifier",
            "Action type",
            "Event type",
            "Event timestamp",
            "Effective Date",
            "Execution Timestamp",
            "Expiration Date",
            "Platform identifier",
            "Notional amount-Leg 1",
            "Notional currency-Leg 1",
            "Fixed rate-Leg 1",
            "Fixed rate-Leg 2",
            "Package transaction price",
            "Other payment amount"
        ]

        new_df = df[selected_columns]
        new_df["Notional amount-Leg 1"] = new_df["Notional amount-Leg 1"].fillna('').astype(str).str.replace('[^\d]', '', regex=True)
        new_df["Notional amount-Leg 1"] = pd.to_numeric(new_df["Notional amount-Leg 1"], errors='coerce').fillna(0).astype('int64')

        for idx, row in new_df.iterrows():
            try:
                other_payment_amount = float(row['Other payment amount'])
                is_nan = math.isnan(other_payment_amount)
            except ValueError:
                is_nan = False 
            if (row["Notional currency-Leg 1"] in ["GBP", "USD", "EUR"]) and is_nan:
                #print(row)
                eff_date = pd.to_datetime(row["Effective Date"])
                exp_date = pd.to_datetime(row["Expiration Date"])
                #print(eff_date)
                timestamp = row["Execution Timestamp"]
                
                notional = row["Notional amount-Leg 1"]
                rate1 = row['Fixed rate-Leg 1']
                rate2 = row['Fixed rate-Leg 2']

                rate = rate2 if pd.isna(rate1) else rate1
                dv01 = get_dv01(notional, rate, eff_date, exp_date)

                event_type = ""
                flag_path = ""
                if (str(eff_date.date()), str(exp_date.date())) in mpc:
                    event_type = "MPC"
                    flag_path = 'uk_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in fomc:
                    event_type = "FOMC"
                    flag_path = 'us_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in ecb:
                    event_type = "ECB"
                    dv01 = dv01 * 1.5
                    flag_path = 'eu_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in cad:
                    event_type = "BoC"
                    flag_path = 'cad_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in aud:
                    event_type = "RBA"
                    flag_path = 'aud_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in nzd:
                    event_type = "RBNZ"
                    flag_path = 'nzd_flag.png'  # Use the flag image
                elif (str(eff_date.date()), str(exp_date.date())) in jpy:
                    event_type = "BoJ"
                    flag_path = 'jpy_flag.png'  # Use the flag image

                
                dv01 = float(f"{dv01:.2g}")
                if event_type:
                    print("timestamp: ",timestamp)
                    print(row)
                    send_notification(event_type, eff_date, rate, row["Notional currency-Leg 1"], dv01, timestamp)
                    print(f"{event_type} detected: DV01 = {dv01}")