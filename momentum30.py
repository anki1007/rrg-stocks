import os
import webbrowser
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

import pandas as pd
import numpy as np
import yfinance as yf

# ================== CONFIG ==================
# CSV universe files (from your folder)
CSV_FILES = {
    "Nifty 200":             r"D:\RRG\ticker\nifty200.csv",
    "Nifty 500":             r"D:\RRG\ticker\nifty500.csv",
    "Nifty Midcap 150":      r"D:\RRG\ticker\niftymidcap150.csv",
    "Nifty Mid+Small 400":   r"D:\RRG\ticker\niftymidsmallcap400.csv",
    "Nifty Smallcap 250":    r"D:\RRG\ticker\niftysmallcap250.csv",
    "Nifty Total Market":    r"D:\RRG\ticker\niftytotalmarket.csv",
}

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}

# Data windows
MOMENTUM_YEARS = 2           # download window for momentum screen
RS_LOOKBACK_DAYS = 252       # ~1y for RS calculations
JDK_WINDOW = 21              # JdK standard window

# ================== HELPERS ==================
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df, symbol: str) -> pd.Series:
    if isinstance(df, pd.Series):
        return df.dropna()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in ("Close", "Adj Close"):
                if (symbol, lvl) in df.columns:
                    return df[(symbol, lvl)].dropna()
        else:
            for col in ("Close", "Adj Close"):
                if col in df.columns:
                    return df[col].dropna()
    return pd.Series(dtype=float)

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"] / df["b"])              # raw RS line
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()   # ~centered around 100
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc - m2) / s2).dropna() # ~centered around 101
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def perf_quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series) -> dict | None:
    """Script-1 filter + compute 6M/3M/1M returns."""
    if adj is None or adj.empty or len(adj) < 252:
        return None
    ema100 = adj.ewm(span=100, adjust=False).mean()
    try:
        one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1.0) * 100.0
    except Exception:
        return None
    high_52w = adj.iloc[-252:].max()
    within_20pct_high = adj.iloc[-1] >= high_52w * 0.8
    if len(adj) < 126:
        return None
    six_month = adj.iloc[-126:]
    up_days_pct = (six_month.pct_change() > 0).sum() / len(six_month) * 100.0
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5 and
        within_20pct_high and up_days_pct > 45.0):
        try:
            r6 = (adj.iloc[-1] / adj.iloc[-126] - 1.0) * 100.0
            r3 = (adj.iloc[-1] / adj.iloc[-63]  - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-21]  - 1.0) * 100.0
        except Exception:
            return None
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
    return None

def load_universe_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["symbol", "company name", "industry"]
    for n in need:
        if n not in cols:
            raise ValueError(f"CSV must include columns: {need}. Missing: {n}")
    df = df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

# ================== DATA BUILD ==================
def build_table_dataframe(benchmark: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    tickers = universe_df["Symbol"].tolist()
    raw = yf.download(
        tickers + [benchmark],
        period=f"{MOMENTUM_YEARS}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    bench = _pick_close(raw, benchmark).dropna()
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark} series empty")
    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()

    rows = []
    for _, rec in universe_df.iterrows():
        sym = rec.Symbol
        name = rec.Name
        industry = rec.Industry
        s = _pick_close(raw, sym).dropna()
        if s.empty:
            continue
        mom = analyze_momentum(s)
        if mom is None:
            continue
        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty:
            continue
        ix = rr.index.intersection(mm.index)
        rr_last = float(rr.loc[ix].iloc[-1])
        mm_last = float(mm.loc[ix].iloc[-1])
        status = perf_quadrant(rr_last, mm_last)
        rows.append({
            "Name": name,
            "Industry": industry,
            "Return_6M": mom["Return_6M"],
            "Return_3M": mom["Return_3M"],
            "Return_1M": mom["Return_1M"],
            "RS-Ratio": rr_last,
            "RS-Momentum": mm_last,
            "Performance": status,
            "Symbol": sym,  # keep for click-through, will be dropped before export
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters.")

    df = pd.DataFrame(rows)
    # rounding
    for col in ("Return_6M", "Return_3M", "Return_1M"):
        df[col] = df[col].round(1)
    df["RS-Ratio"] = df["RS-Ratio"].round(2)
    df["RS-Momentum"] = df["RS-Momentum"].round(2)

    # ranks and position
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")
    df["Final_Rank"] = df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]
    df = df.sort_values("Final_Rank").reset_index(drop=True)
    df["Position"] = np.arange(1, len(df) + 1)
    df.insert(0, "S.No", np.arange(1, len(df) + 1))

    # Final display order (keep Symbol separately for hyperlink only)
    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "Final_Rank", "Position", "Symbol"]
    return df[order]

# ================== UI ==================
HEADERS = [
    "S.No", "Name", "Industry", "Return_6M", "Rank_6M",
    "Return_3M", "Rank_3M", "Return_1M", "Rank_1M",
    "RS-Ratio", "RS-Momentum", "Performance", "Final_Rank", "Position",
]

def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "#dff5df"  # light green
    if sno <= 60: return "#fff6b3"  # light yellow
    if sno <= 90: return "#dfe9ff"  # light blue
    return "#f7d6d6"                # light red

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nifty Total Market Momentum")

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        # Benchmark selection (no custom box)
        ttk.Label(top, text="Benchmark:").pack(side="left")
        self.bench_var = tk.StringVar(value="Nifty 500")
        self.bench_combo = ttk.Combobox(
            top,
            textvariable=self.bench_var,
            values=list(BENCHMARKS.keys()),
            width=24,
            state="readonly"
        )
        self.bench_combo.pack(side="left", padx=6)

        # Universe CSV dropdown
        ttk.Label(top, text="Universe:").pack(side="left", padx=(12, 0))
        self.csv_var = tk.StringVar(value="Nifty Smallcap 250")
        self.csv_combo = ttk.Combobox(
            top,
            textvariable=self.csv_var,
            values=list(CSV_FILES.keys()),
            width=22,
            state="readonly",
        )
        self.csv_combo.pack(side="left", padx=6)

        ttk.Button(top, text="Load / Refresh", command=self.load_async).pack(side="left")
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side="left", padx=6)

        self.status = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status).pack(side="right")

        # Scrollable canvas table
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.table = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.header_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.cell_font = tkfont.Font(family="Segoe UI", size=11)

        self.df = pd.DataFrame()
        self.load_async()

    def current_benchmark(self) -> str:
        return BENCHMARKS.get(self.bench_var.get().strip())

    def load_async(self):
        if getattr(self, "_thread", None) and self._thread.is_alive():
            messagebox.showinfo("Loading", "Fetch already in progress.")
            return
        self.status.set("Loading…")
        self._thread = threading.Thread(target=self._load_data, daemon=True)
        self._thread.start()

    def _load_data(self):
        try:
            # Use selected CSV file
            selected_csv_path = CSV_FILES.get(self.csv_var.get())
            if not selected_csv_path:
                raise RuntimeError("Please select a universe CSV.")
            uni = load_universe_from_csv(selected_csv_path)

            bench = self.current_benchmark()
            if not bench:
                raise RuntimeError("Please pick a valid benchmark")

            df = build_table_dataframe(bench, uni)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status.set(""))
            return
        self.root.after(0, lambda: self._render_table(df))

    def _render_table(self, df: pd.DataFrame):
        for w in self.table.winfo_children():
            w.destroy()

        PAD_PX = 18
        col_px = [self.header_font.measure(h) + PAD_PX for h in HEADERS]

        def row_vals(r):
            return [
                int(r["S.No"]), str(r["Name"]), str(r["Industry"]),
                f"{r['Return_6M']:.1f}", f"{r['Rank_6M']:.0f}",
                f"{r['Return_3M']:.1f}", f"{r['Rank_3M']:.0f}",
                f"{r['Return_1M']:.1f}", f"{r['Rank_1M']:.0f}",
                f"{r['RS-Ratio']:.2f}", f"{r['RS-Momentum']:.2f}",
                str(r["Performance"]), f"{r['Final_Rank']:.0f}", f"{r['Position']:.0f}",
            ]

        for _, r in df.iterrows():
            vals = row_vals(r)
            for j, val in enumerate(vals):
                w = self.cell_font.measure(str(val)) + PAD_PX
                if w > col_px[j]:
                    col_px[j] = w

        # headers
        for j, h in enumerate(HEADERS):
            anchor = "w" if h in ("Name", "Industry") else "center"
            tk.Label(self.table, text=h, relief=tk.RIDGE, font=self.header_font, anchor=anchor, bg="#ececec")\
              .grid(row=0, column=j, sticky="nsew", padx=1, pady=1)
            self.table.grid_columnconfigure(j, minsize=int(col_px[j]))
            self.table.grid_columnconfigure(j, weight=(3 if j in (1, 2) else 1), uniform="cols")

        # rows
        for i, r in df.iterrows():
            sno = int(r["S.No"])
            bg = row_bg_for_serial(sno)
            vals = row_vals(r)
            for j, val in enumerate(vals):
                anchor = "w" if j in (1, 2) else "center"
                lbl = tk.Label(self.table, text=str(val), relief=tk.RIDGE, font=self.cell_font, bg=bg, anchor=anchor)
                lbl.grid(row=i+1, column=j, sticky="nsew", padx=1, pady=1)
                if j == 1:  # Name clickable
                    sym = str(r.get("Symbol", "")).strip()
                    if sym:
                        lbl.bind("<Button-1>", lambda e, s=sym: webbrowser.open_new_tab(tradingview_chart_url(s)))
                        lbl.configure(cursor="hand2")

        # store df for export (drop Symbol)
        self.df = df.drop(columns=["Symbol"]).copy()
        self.status.set(f"Rows: {len(df)}")

    def export_csv(self):
        if self.df.empty:
            messagebox.showwarning("Export", "Nothing to export.")
            return
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Base the filename on the selected CSV file
        selected_path = CSV_FILES.get(self.csv_var.get())
        base = os.path.splitext(os.path.basename(selected_path))[0] if selected_path else "universe"

        out = os.path.join(os.path.expanduser("~"), "Downloads", f"{base}_momentum_{ts}.csv")
        try:
            self.df.to_csv(out, index=False)
            messagebox.showinfo("Export", f"Saved {out}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save CSV.\n\n{e}")

if __name__ == "__main__":
    app = App()
    app.root.geometry("1300x800")
    app.root.mainloop()
