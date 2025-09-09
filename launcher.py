import threading
import subprocess
import webview

def run_flask():
    subprocess.run(['python', 'app.py'])

threading.Thread(target=run_flask).start()
webview.create_window("Invoice & Accounting AI Assistant", "http://127.0.0.1:5000")
webview.start()