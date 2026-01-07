from flask import Flask, render_template
import threading
from smart_board import start_smartboard

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    t = threading.Thread(target=start_smartboard)
    t.daemon = True  # allows Flask to continue
    t.start()
    return "Smart Board Started"

