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

locale.setlocale(locale.LC_ALL, 'en_US')

base_url = r'https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/slices/CFTC_SLICE_RATES_{:%Y_%m_%d}_{}.zip'

mpc = {("2024-06-20", "2024-08-01"), ("2024-08-01", "2024-09-19"), ("2024-09-19", "2024-11-07"), ("2024-11-07", "2024-12-19")}
fomc = {("2024-06-12", "2024-07-31"), ("2024-07-31", "2024-09-18"), ("2024-09-18", "2024-11-07"), ("2024-11-07", "2024-12-18")}
ecb = {("2024-06-12", "2024-07-24"), ("2024-07-24", "2024-09-18"), ("2024-09-18", "2024-10-23"), ("2024-10-23", "2024-12-18")}

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
        raise Exception('Problem downloading stuff')
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
    dv01_text = f"{currency_symbol}{dv01:.3f}m"
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
        root.destroy()
        os.remove(temp_image_path)

    root = Tk()
    root.overrideredirect(True)
    root.geometry(f"{image.width}x{image.height}+{root.winfo_screenwidth() - image.width - 10}+{root.winfo_screenheight() - image.height - 50}")

    # Convert the PIL image to a format Tkinter can use
    tk_image = ImageTk.PhotoImage(image)

    label = Label(root, image=tk_image)
    label.image = tk_image  # keep a reference!
    label.pack()

    threading.Thread(target=close_window).start()
    root.mainloop()

def send_notification(event_type, effective_date, rate, currency, dv01, timestamp, logo_path, flag_path):
    if dv01 > 10:  # Only show notification if DV01 is greater than 10
        image = create_image(effective_date, event_type, rate, currency, dv01, timestamp, logo_path, flag_path)
        # Start the notification in a new thread to avoid blocking
        threading.Thread(target=show_notification, args=(image,)).start()

ref_date = datetime.date.today() 
#counter = 800
flag = False
for counter in range(1050, 1100):
    print("counter: ", counter)
    try:
        if flag == True:
            break
        zip_content = get_dtcc_file(ref_date, counter)

        with ZipFile(BytesIO(zip_content)) as z:
            file_list = z.namelist()
            
            with z.open(file_list[0]) as f:
                df = pd.read_csv(f)
                #df.to_csv('attempt2.csv')

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
                ]

                new_df = df[selected_columns]
                new_df["Notional amount-Leg 1"] = new_df["Notional amount-Leg 1"].fillna('').astype(str).str.replace('[^\d]', '', regex=True)
                new_df["Notional amount-Leg 1"] = pd.to_numeric(new_df["Notional amount-Leg 1"], errors='coerce').fillna(0).astype('int64')

                
                for idx, row in new_df.iterrows():
                    if row["Dissemination Identifier"]==1046743784:
                        #print(row['Dissemination Identifier'])
                        print("found it: ",counter)
                        flag = True
                        break

        
                        
    except Exception as e:
        print(e)
        print(f"Erro: e <GO>")
