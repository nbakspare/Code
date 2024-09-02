import datetime
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import locale
from dateutil.relativedelta import relativedelta
from PIL import Image, ImageDraw, ImageFont, ImageTk
import threading
import os
from tkinter import Tk, Label
import time
import math

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

def create_image(effective_date, event_type, rate, currency, dv01, timestamp, logo_path, flag_path):
    # Load the logo image
    logo = Image.open(logo_path)
    logo = logo.resize((200, 80))  # Resize logo as needed

    # Load the flag image
    flag = Image.open(flag_path)
    flag = flag.resize((30, 30))  # Resize flag as needed

    # Define image size and create a new image
    width, height = 600, 240  # Reduced height for less white space
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Define fonts
    font_path = "times.ttf"  # Change to a more professional font
    font = ImageFont.truetype(font_path, 24)
    bold_font = ImageFont.truetype(font_path, 24)

    # Define text parts
    rate_percent = round(rate * 100, 3)  # Convert rate to percentage and round to 3sf
    text_line1_parts = [
        (effective_date.strftime('%B'), "red"),
        (f" {event_type} trades at ", "black"),
        (f"{rate_percent}%", "green")
    ]
    currency_symbol = "£" if currency == "GBP" else ("$" if currency == "USD" else "€")
    dv01_text = f"{currency_symbol}{dv01:.3f}k"
    text_line2_parts = [
        ("with DV01 ", "black"),
        (dv01_text, "blue")
    ]
    timestamp_dt = pd.to_datetime(timestamp)
    timestamp_dt_plus_one_hour = timestamp_dt + pd.Timedelta(hours=1)

    # Format the datetime object and extract the desired part
    timestamp_text = f"Time Stamp: {timestamp_dt_plus_one_hour.strftime('%Y-%m-%d %H:%M:%S')[-9:-1]}"

    # Calculate text positions
    text_x = 60
    text_y1 = 10
    text_y2 = text_y1 + 25  # Reduced space between first and second line
    text_y3 = text_y2 + 50  # Increased space between second and third line

    # Draw flag
    image.paste(flag, (int(text_x - 50), int(text_y1)), flag.convert("RGBA"))

    # Draw text parts for the first line
    for part, color in text_line1_parts:
        part_str = str(part)  # Ensure the part is a string
        draw.text((text_x, text_y1), part_str, fill=color, font=font if color == "black" else bold_font)
        text_x += draw.textlength(part_str, font=font if color == "black" else bold_font)

    # Reset text_x for the second line
    text_x = 60

    # Draw text parts for the second line
    for part, color in text_line2_parts:
        part_str = str(part)  # Ensure the part is a string
        draw.text((text_x, text_y2), part_str, fill=color, font=font if color == "black" else bold_font)
        text_x += draw.textlength(part_str, font=font if color == "black" else bold_font)

    # Draw the timestamp as the third line, centered and in bold
    timestamp_w = draw.textlength(timestamp_text, font=bold_font)
    timestamp_x = (width - timestamp_w) / 2
    draw.text((timestamp_x, text_y3), timestamp_text, fill="black", font=bold_font)

    # Paste the logo closer to the text
    logo_x = (width - logo.width) / 2
    logo_y = text_y3 + 30  # Reduced space below the timestamp
    image.paste(logo, (int(logo_x), logo_y), logo.convert("RGBA"))

    return image

def show_notification(image):
    temp_image_path = "notification.png"
    image.save(temp_image_path)

    def close_window():
        time.sleep(30)  # Increase display time to 30 seconds
        if root.winfo_exists():  # Check if the window exists before destroying
            root.destroy()
        os.remove(temp_image_path)

    root = Tk()
    root.overrideredirect(True)
    root.geometry(f"{image.width}x{image.height}+{root.winfo_screenwidth() - image.width - 10}+{root.winfo_screenheight() - image.height - 50}")

    # Convert the PIL image to a format Tkinter can use
    tk_image = ImageTk.PhotoImage(image)

    label = Label(root, image=tk_image)
    label.image = tk_image  # keep a reference to avoid garbage collection
    label.pack()

    threading.Thread(target=close_window).start()
    root.mainloop()

# Ensure the Tk application is properly referenced and cleaned up
def send_notification(event_type, effective_date, rate, currency, dv01, timestamp, logo_path, flag_path):
    if dv01 > 10:  # Only show notification if DV01 is greater than 10
        image = create_image(effective_date, event_type, rate, currency, dv01, timestamp, logo_path, flag_path)
        # Start the notification and wait for it to close before proceeding
        show_notification(image)

ref_date = datetime.date.today() #- datetime.timedelta(days=3)
counter = 1576

try:
    while True:
        print(f"Processing counter: {counter}")
        zip_content = get_dtcc_file(ref_date, counter)
        
        if zip_content is None:
            time.sleep(60)
            continue

        with ZipFile(BytesIO(zip_content)) as z:
            file_list = z.namelist()
            
            with z.open(file_list[0]) as f:
                df = pd.read_csv(f)
                #df.to_csv(f'attempt2_counter_{counter}.csv')

                selected_columns = [
                    "Dissemination Identifier",
                    "Action type",
                    "Event type",
                    "Event timestamp",
                    "Effective Date",
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
                        eff_date = pd.to_datetime(row["Effective Date"])
                        exp_date = pd.to_datetime(row["Expiration Date"])
                        timestamp = row["Event timestamp"]
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
                            dv01= dv01*1.5
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
                            send_notification(event_type, eff_date, rate, row["Notional currency-Leg 1"], dv01, timestamp, 'coex.png', flag_path)
                            print(f"{event_type} detected: DV01 = {dv01}")

        counter += 1

except Exception as e:
    print(e)
    print(f"Erro: e <GO>")
