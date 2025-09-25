from flask import Flask, render_template, request, send_from_directory, jsonify
from fpdf import FPDF
import os
import pandas as pd
from datetime import datetime
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from threading import Timer
import webbrowser
import requests

app = Flask(__name__)
INVOICE_DIR = 'invoices'
DATA_FILE = 'data/invoices_data.csv'
GUMROAD_PRODUCT_ID = "UPgBmp1Q1kpOG5H3ETLxyA=="

os.makedirs(INVOICE_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=['InvoiceID', 'Customer', 'Amount', 'Currency', 'Date']).to_csv(DATA_FILE, index=False)

CURRENCY_SYMBOLS = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', 'A$': 'AUD', 'C$': 'CAD', 'CHF': 'CHF'}


def verify_license(license_key):
    license_key = license_key.strip()  # Remove leading/trailing spaces
    if os.environ.get("APP_ENV") == "development":
        return True
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {"product_id": GUMROAD_PRODUCT_ID, "license_key": license_key}
    try:
        response = requests.post(url, data=payload, timeout=5)
        data = response.json()

        # DEBUG: log the full response for troubleshooting
        print("Gumroad response:", data)

        # Gumroad returns success in two places sometimes
        if data.get("success") is True:
            return True
        elif "purchase" in data and data["purchase"].get("license") == license_key:
            return True
        return False
    except Exception as e:
        print("License verification error:", e)
        return False

def parse_nl_invoice(text):
    customer_match = re.search(r'for (.+)', text)
    customer = customer_match.group(1).strip() if customer_match else 'Unknown'

    pattern = r'(\d+)\s+([a-zA-Z ]+)\s+at\s+\$?(\d+\.?\d*)'
    items = []
    for m in re.findall(pattern, text):
        qty = int(m[0])
        name = m[1].strip()
        price = float(m[2])
        items.append((name, qty, price))
    return customer, items


def predict_revenue(df):
    numeric_df = df[pd.to_numeric(df['Amount'], errors='coerce').notnull()]
    if numeric_df.shape[0] < 3:
        return None  # Not enough data yet

    # Convert dates to timestamps
    numeric_df['Timestamp'] = pd.to_datetime(numeric_df['Date']).map(datetime.timestamp)

    # Normalize timestamps to start at zero
    min_timestamp = numeric_df['Timestamp'].min()
    numeric_df['Timestamp'] -= min_timestamp

    X = numeric_df[['Timestamp']].values.reshape(-1, 1)
    y = numeric_df['Amount'].values

    model = LinearRegression().fit(X, y)

    # Predict 30 days after the last invoice, normalized same as training
    last_timestamp = numeric_df['Timestamp'].max()
    next_time = np.array([[last_timestamp + 30 * 24 * 3600]])  # 30 days later
    pred = model.predict(next_time)[0]

    return round(max(float(pred), 0), 2)


def apply_smart_discount(parsed_items, total):
    discount = 0
    suggestions = []

    for name, qty, price in parsed_items:
        if qty >= 50:
            bulk_discount = qty * price * 0.10
            discount += bulk_discount
            suggestions.append(f"Bulk discount on {name}: 10% off = {bulk_discount:.2f}")
            
    if total > 2000:
        tier_discount = total * 0.15
        discount += tier_discount
        suggestions.append(f"Tiered discount: 15% off = {tier_discount:.2f}")
    elif total > 1000:
        tier_discount = total * 0.10
        discount += tier_discount
        suggestions.append(f"Tiered discount: 10% off = {tier_discount:.2f}")
    elif total > 500:
        tier_discount = total * 0.05
        discount += tier_discount
        suggestions.append(f"Tiered discount: 5% off = {tier_discount:.2f}")
        
    rounded_total = round((total - discount) / 10) * 10
    rounding_discount = (total - discount) - rounded_total
    if rounding_discount > 0:
        discount += rounding_discount
        suggestions.append(f"Rounding discount: {rounding_discount:.2f}")
    return total - discount, discount, suggestions

@app.route('/verify_license', methods=['POST'])
def verify_license_route():
    data = request.get_json()
    license_key = data.get("license")
    valid = verify_license(license_key)
    return jsonify({"success": valid})
    

@app.route('/', methods=['GET', 'POST'])
def index():
    invoice_filename = None
    invoice_url = None
    suggestions = ''
    predicted_revenue = None
    license_verified = False  

    df = pd.read_csv(DATA_FILE)

    if request.method == 'POST':
        license_key = request.form.get("license")
        if license_key and verify_license(license_key):
            license_verified = True 
        else:
            return "Invalid license key. Please enter a valid license.", 403

        use_nl = request.form.get('use_nl')
        items_input = request.form.get('items')
        apply_discount = request.form.get('apply_discount')
        currency = request.form.get('currency', 'USD')
        symbol = CURRENCY_SYMBOLS.get(currency, '$')

        if use_nl:
            customer, parsed_items = parse_nl_invoice(items_input)
        else:
            customer = request.form.get('customer')
            parsed_items = []
            for line in items_input.splitlines():
                try:
                    name, qty, price = line.split(',')
                    parsed_items.append((name.strip(), int(qty.strip()), float(price.strip())))
                except:
                    continue

        # Discounts
        total = sum(qty * price for _, qty, price in parsed_items)
        suggestions = ""
        selected_discounts = request.form.getlist('discounts')
        # Apply threshold, tiered, bulk, rounding discounts here
        if "threshold" in selected_discounts:
            threshold_option = request.form.get('threshold_option')
            percent, limit = map(float, threshold_option.split('-'))
            if total > limit:
                discount_amount = total * (percent / 100)
                total -= discount_amount
                suggestions += f"Threshold discount applied: {percent}% off = {discount_amount:.2f}\n"
        if "tiered" in selected_discounts:
            tier_discount = 0
            if total > 2000:
                tier_discount = total * 0.15
            elif total > 1000:
                tier_discount = total * 0.10
            elif total > 500:
                tier_discount = total * 0.05
            if tier_discount > 0:
                total -= tier_discount
                suggestions += f"Tiered discount applied: -{tier_discount:.2f}\n"
        if "bulk" in selected_discounts:
            bulk_discount = sum(qty*price*0.10 for _, qty, price in parsed_items if qty>=50)
            if bulk_discount > 0:
                total -= bulk_discount
                suggestions += f"Bulk discount applied: -{bulk_discount:.2f}\n"
        if "rounding" in selected_discounts:
            rounded_total = round(total, -1) - 1
            rounding_discount = total - rounded_total
            if rounding_discount > 0:
                total = rounded_total
                suggestions += f"Smart rounding applied: -{rounding_discount:.2f}\n"

        predicted_revenue = predict_revenue(df)
        invoice_id = f"INV-{int(datetime.now().timestamp())}"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Invoice ID: {invoice_id}", ln=True)
        pdf.cell(200, 10, f"Customer: {customer}", ln=True)
        pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
        pdf.cell(200, 10, f"Currency: {currency} ({symbol})", ln=True)
        pdf.ln(5)
        pdf.cell(50, 10, "Item", border=1)
        pdf.cell(30, 10, "Qty", border=1)
        pdf.cell(30, 10, "Price", border=1)
        pdf.cell(30, 10, "Total", border=1)
        pdf.ln()
        for name, qty, price in parsed_items:
            pdf.cell(50, 10, name, border=1)
            pdf.cell(30, 10, str(qty), border=1)
            pdf.cell(30, 10, f"{symbol} {price:.2f}", border=1)
            pdf.cell(30, 10, f"{symbol} {qty*price:.2f}", border=1)
            pdf.ln()
        pdf.ln(5)
        pdf.cell(200, 10, f"Total Amount: {symbol} {total:.2f}", ln=True)

        invoice_filename = f"{invoice_id}.pdf"
        invoice_path = os.path.join(INVOICE_DIR, invoice_filename)
        pdf.output(invoice_path)

        df = pd.concat([df, pd.DataFrame([{
            "InvoiceID": invoice_id,
            "Customer": customer,
            "Amount": total,
            "Currency": currency,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        invoice_url = f"{request.url_root}invoices/{invoice_filename}"

    return render_template(
        'index.html',
        invoice_filename=invoice_filename,
        invoice_url=invoice_url,
        suggestions=suggestions,
        predicted_revenue=predicted_revenue,
        currencies=CURRENCY_SYMBOLS.keys(),
        license_verified=license_verified
    )

@app.route('/invoices/<filename>')
def download_invoice(filename):
        
    path = os.path.join(INVOICE_DIR, filename)
    if not os.path.exists(path):
        return "Invoice not found", 404
    return send_from_directory(os.path.abspath(INVOICE_DIR), filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)