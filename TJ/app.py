"""
app.py
------

Main entry point for the Flask web application that provides a user
interface for managing trades. The UI allows traders to manually
input trades, view their trade history, compute performance metrics
and export data. The application uses modular components defined
elsewhere in the package (models, database, analytics) to separate
business logic from presentation.

To run the application:
    1. Ensure dependencies are installed (see requirements.txt).
    2. Execute `python app.py` from within the tradezella_app directory.
    3. Navigate to http://localhost:5000 in your web browser.

Note: The Flask development server is intended for local use. For
production deployments consider using a production WSGI server
such as Gunicorn.
"""
import os
import io
import csv
from datetime import datetime
import requests

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, Response
)
from werkzeug.utils import secure_filename  # optional; safe filename

from .database import TradeJournalDB
from .analytics import compute_metrics
from .models import Trade

ALLOWED_CSV = {"csv"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_CSV
def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 500):
    """
    Fetch klines from Binance and map to Lightweight Charts shape.
    Returns a list[dict]: { time (sec), open, high, low, close, volume }
    """
    # Binance uses uppercase symbols like BTCUSDT
    symbol = symbol.upper().strip()
    limit = max(1, min(int(limit), 1000))

    # Binance kline intervals: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()

    # Map Binance array format → dict format expected by your front-end
    # Each kline: [ openTime, open, high, low, close, volume, closeTime, ... ]
    out = []
    for k in raw:
        open_time_ms = k[0]
        out.append({
            "time": open_time_ms // 1000,     # seconds
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    # Ensure ascending order by time (Binance already is, but keep safe)
    out.sort(key=lambda x: x["time"])
    return out
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    db_path = os.getenv("TJ_DB", "tradezella.db")
    db = TradeJournalDB(db_path)

    # ---------- routes ----------
    @app.route("/export", methods=["GET"], endpoint="export")
    def export_trades():
        trades = db.list_trades()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id","instrument","date_time","position","entry_price","exit_price","quantity","notes"])
        for t in trades:
            w.writerow([
                t.id,
                t.instrument,
                t.date_time.isoformat(),
                t.position,
                t.entry_price,
                t.exit_price,
                t.quantity,
                (t.notes or "").replace("\n", " ").strip(),
            ])
        out.seek(0)
        return Response(
            out.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=trades.csv"},
        )

    @app.route("/")
    def index():
        trades = db.list_trades()
        m = compute_metrics(trades)   # ← add this
        return render_template("index.html", title="Trade History", trades=trades, metrics=m)

    @app.route("/add", methods=["GET", "POST"])
    def add_trade():
        if request.method == "POST":
            instrument = request.form["instrument"].strip()
            dt_str = request.form["date_time"].strip().replace("T", " ")
            date_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            position = request.form["position"].strip().lower()  # long/short
            entry_price = float(request.form["entry_price"])
            exit_price = float(request.form["exit_price"])
            quantity = float(request.form["quantity"])
            notes = request.form.get("notes", "")

            t = Trade(
                id=None,
                instrument=instrument,
                date_time=date_time,
                position=position,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                notes=notes,
            )
            db.add_trade(t)
            flash("Trade added.", "success")
            return redirect(url_for("index"))

        return render_template("add_trade.html", title="Add Trade")

    @app.route("/metrics", methods=["GET", "POST"])
    def metrics():
        if request.method == "POST":
            start = request.form.get("start_date", "").strip()
            end = request.form.get("end_date", "").strip()
            if start and end:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59)
                trades = db.trades_between(start_dt, end_dt)
            else:
                trades = db.list_trades()
        else:
            trades = db.list_trades()

        m = compute_metrics(trades)
        return render_template("metrics.html", title="Metrics", metrics=m)

    # -------- OHLC UPLOAD / API / CHART --------

    @app.route("/upload-ohlc", methods=["GET", "POST"])
    def upload_ohlc():
        if request.method == "POST":
            f = request.files.get("file")
            if not f or f.filename == "":
                flash("Please choose a CSV file.", "error")
                return redirect(url_for("upload_ohlc"))

            if not allowed_file(f.filename):
                flash("Only .csv files are supported.", "error")
                return redirect(url_for("upload_ohlc"))

            content = f.read().decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(content))
            fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]

            # Required columns
            required = {"open", "high", "low", "close", "symbol"}
            missing = required - set(fieldnames)
            if "timestamp" not in fieldnames and "ts" not in fieldnames:
                missing.add("timestamp")
            if missing:
                flash(f"Missing columns: {', '.join(sorted(missing))}", "error")
                return redirect(url_for("upload_ohlc"))

            rows = []
            for row in reader:
                r = {k.lower(): v for k, v in row.items()}
                ts = r.get("timestamp") or r.get("ts")
                rows.append(
                    {
                        "symbol": r["symbol"],
                        "ts": int(float(ts)),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r.get("volume", 0) or 0),
                    }
                )
            inserted = db.upsert_ohlc_rows(rows)
            flash(f"Uploaded/updated {inserted} OHLC rows.", "success")
            return redirect(url_for("chart"))

        symbols = db.get_symbols()
        return render_template("upload_ohlc.html", title="Upload OHLC", symbols=symbols)

    @app.route("/api/ohlc")
    def api_ohlc():
        symbol = (request.args.get("symbol") or "BTCUSDT").upper().strip()
        limit = int(request.args.get("limit", 2000))
        interval = request.args.get("interval", "1h").strip()
        source = request.args.get("source", "binance").strip().lower()
        # source ∈ {"binance", "db"}

        if source == "binance":
            try:
                data = fetch_binance_klines(symbol, interval=interval, limit=limit)
                return jsonify(data)
            except requests.HTTPError as e:
                return jsonify({"error": f"Binance API error: {e}"}), 502
            except Exception as e:
                return jsonify({"error": f"Upstream fetch failed: {e}"}), 502

        # fallback to DB mode (your original behavior)
        if not symbol:
            return jsonify({"error": "symbol query param required"}), 400
        data = db.fetch_ohlc(symbol, limit=limit)
        # If your DB stores ms timestamps, convert here for consistency:
        # for d in data: d["time"] = d.pop("ts") // 1000
        return jsonify(data)
    @app.route("/chart")
    def chart():
        symbols = db.get_symbols()
        selected = request.args.get("symbol") or (symbols[0] if symbols else "BTCUSDT")
        return render_template("chart.html", title="Chart", symbols=symbols or ["BTCUSDT","ETHUSDT"], selected=selected)




    
    # @app.route("/chart")
    # def chart():
    #     symbols = db.get_symbols()
    #     selected = request.args.get("symbol") or (symbols[0] if symbols else "")
    #     return render_template("chart.html", title="Chart", symbols=symbols, selected=selected)

    # @app.route("/api/ohlc")
    # def api_ohlc():
    #     symbol = (request.args.get("symbol") or "").upper().strip()
    #     limit = int(request.args.get("limit", 1000))
    #     if not symbol:
    #         return jsonify({"error": "symbol query param required"}), 400
    #     data = db.fetch_ohlc(symbol, limit=limit)
    #     return jsonify(data)
   

    return app


# Run directly
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5004, debug=True, use_reloader=False)
