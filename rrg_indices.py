# --- TABLE (scrollable + larger fonts + collapse/expand + visible links) ---
st.markdown(f"**Date:** {str(idx[sel_pos].date())}")

def fmt_num(x, n=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "-"
    return f"{x:,.{n}f}"

# Build rows
rows = []
for t in tickers:
    try:
        rr = float(rs_ratio_map[t].iloc[sel_pos]); mm = float(rs_mom_map[t].iloc[sel_pos])
        px = px_cache.get(t, pd.Series(dtype=float))
        price = float(px.iloc[sel_pos]) if sel_pos < len(px) else np.nan
        chg = ((px.iloc[sel_pos]/px.iloc[start_pos]-1)*100.0) if (sel_pos < len(px) and start_pos < len(px)) else np.nan
        status = get_status(rr, mm)
        rows.append((t, display_name(t), status, price, chg))
    except Exception:
        pass

# Ranking → SL No.
all_perf = []
for t in tickers:
    try:
        all_perf.append((t, compute_rank_metric(t, start_pos, sel_pos)))
    except Exception:
        pass
all_perf.sort(key=lambda x: x[1], reverse=True)
rank_map = {sym:i for i,(sym,_m) in enumerate(all_perf, start=1)}

# Build HTML with larger fonts + scrollbar container
table_css = """
<style>
.rrg-wrap { max-height: 520px; overflow-y: auto; border: 1px solid rgba(128,128,128,0.25); border-radius: 6px; }
.rrg-table { width:100%; border-collapse:collapse; }
.rrg-table th, .rrg-table td {
  padding:10px 12px; border-bottom:1px solid rgba(128,128,128,0.22);
  font-family: Segoe UI, Inter, Arial, system-ui; font-size:15px; line-height: 1.2;
}
.rrg-table th { position: sticky; top: 0; background: rgba(200,200,200,0.10); text-align:left; z-index: 1; }
.rrg-color { width:14px; height:14px; border-radius:3px; display:inline-block; border:1px solid rgba(0,0,0,0.25); margin-right:4px; vertical-align: -2px; }
.rrg-name { font-weight: 600; text-decoration: underline; }
</style>
"""

html = [table_css, """
<div class="rrg-wrap">
<table class="rrg-table">
<thead><tr>
<th style="width:72px;">SL No.</th>
<th style="width:26px;"></th>
<th>Name</th>
<th style="width:130px;">Status</th>
<th style="width:140px;">Price</th>
<th style="width:140px;">Change %</th>
</tr></thead>
<tbody>
"""]

for sym, name, status, price, chg in rows:
    sl = rank_map.get(sym, "")
    _, url = tv_build_link(sym)
    rr_last = float(rs_ratio_map[sym].iloc[sel_pos]); mm_last = float(rs_mom_map[sym].iloc[sel_pos])
    bg = status_bg_color(rr_last, mm_last)
    fg = "white" if bg in ("#e06a6a","#3fa46a","#5d86d1") else "black"
    strip = COLORS.get(sym, "#888")
    html.append(
        f'<tr style="background:{bg}; color:{fg};">'
        f'<td>{sl}</td>'
        f'<td><span class="rrg-color" style="background:{strip}"></span></td>'
        # ensure link is visible on colored rows: force link color to row fg
        f'<td><a class="rrg-name" href="{url}" target="_blank" rel="noopener noreferrer" style="color:{fg};">{name}</a></td>'
        f'<td>{status}</td>'
        f'<td>{fmt_num(price,2)}</td>'
        f'<td>{fmt_num(chg,2)}</td>'
        f'</tr>'
    )

html.append("</tbody></table></div>")

with st.expander("Table", expanded=True):   # collapse / expand control
    st.markdown("".join(html), unsafe_allow_html=True)
