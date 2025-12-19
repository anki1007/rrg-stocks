"""
Interactive Table Component for Streamlit
Features: Sorting, Searching, Filtering by Status and Industry
Perfect for RRG Indices Dashboard

BLOOMBERG TERMINAL THEME:
- Sidebar: Dark Blue (#1a3a52)
- Main Page: Beige (#F5F5DC)
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOOMBERG THEME CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_bloomberg_theme():
    """Apply Bloomberg Terminal color scheme to Streamlit app"""
    st.markdown("""
    <style>
    /* Main container - Beige background */
    .main, [data-testid="stAppViewContainer"] {
        background-color: #F5F5DC !important;
    }
    
    /* Sidebar - Dark Bloomberg Blue */
    [data-testid="stSidebar"] {
        background-color: #1a3a52 !important;
    }
    
    /* Sidebar text - Light for contrast */
    [data-testid="stSidebar"] * {
        color: #E8E8E8 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #C0C0C0 !important;
        font-weight: 600;
    }
    
    /* Body text - Dark blue on beige */
    body {
        background-color: #F5F5DC !important;
        color: #1a3a52 !important;
    }
    
    /* Headings - Dark blue */
    h1, h2, h3, h4, h5, h6 {
        color: #1a3a52 !important;
        font-weight: 700;
    }
    
    /* Markdown text */
    .st-emotion-cache-uf99v0 {
        color: #1a3a52 !important;
    }
    
    /* Input fields - White background, dark blue border */
    input[type="text"],
    input[type="number"],
    select,
    textarea {
        background-color: #FFFFFF !important;
        color: #1a3a52 !important;
        border: 2px solid #1a3a52 !important;
        border-radius: 4px;
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] {
        color: #1a3a52 !important;
    }
    
    /* Button - Bloomberg blue background */
    button {
        background-color: #1a3a52 !important;
        color: #F5F5DC !important;
        font-weight: 600;
        border: none !important;
        border-radius: 4px;
    }
    
    button:hover {
        background-color: #0d1f2d !important;
        color: #FFFFFF !important;
    }
    
    /* Dataframe - White background */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }
    
    /* Info/Warning boxes */
    .st-emotion-cache-1l269bu,
    [data-testid="stAlert"] {
        background-color: #FFFACD !important;
        border: 1px solid #1a3a52 !important;
        border-left: 4px solid #1a3a52 !important;
    }
    
    /* Divider - Bloomberg blue */
    hr {
        border-color: #1a3a52 !important;
        border-width: 2px;
    }
    
    /* Links - Bloomberg blue */
    a {
        color: #1a3a52 !important;
        text-decoration: underline;
    }
    
    a:hover {
        color: #0d1f2d !important;
    }
    
    /* Metric boxes */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #1a3a52;
    }
    </style>
    """, unsafe_allow_html=True)


def render_interactive_table(
    df: pd.DataFrame,
    key_prefix: str = "table",
    show_status_filter: bool = True,
    show_industry_filter: bool = True,
    sortable_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Renders an interactive table with sorting, searching, and filtering.
    With Bloomberg Terminal color scheme.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to display
    key_prefix : str
        Unique prefix for session state keys
    show_status_filter : bool
        Show status filter (Lagging, Leading, Improving, Weakening)
    show_industry_filter : bool
        Show industry/category filter
    sortable_columns : List[str], optional
        Columns that are sortable. If None, all numeric columns are sortable.
    numeric_columns : List[str], optional
        Columns to format as numeric. Auto-detected if None.
    
    Returns:
    --------
    pd.DataFrame
        Filtered and sorted DataFrame
    """
    
    if df.empty:
        st.info("📭 No data to display")
        return df
    
    # Detect numeric columns if not provided
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if sortable_columns is None:
        sortable_columns = numeric_columns + df.select_dtypes(include=['object']).columns.tolist()[:3]
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONTROLS: Search, Sort, Filters
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    st.markdown("### 🔍 Table Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    # Search Box
    search_key = f"{key_prefix}_search"
    with col1:
        search_term = st.text_input(
            "🔍 Search",
            value=st.session_state.get(search_key, ""),
            placeholder="Search any column...",
            key=search_key,
        ).lower()
    
    # Sort Column
    sort_key = f"{key_prefix}_sort_col"
    with col2:
        sort_col = st.selectbox(
            "📊 Sort by",
            ["None"] + sortable_columns,
            index=st.session_state.get(f"{sort_key}_idx", 0),
            key=f"{sort_key}_select",
        )
        st.session_state[f"{sort_key}_idx"] = ["None"] + sortable_columns
        st.session_state[f"{sort_key}_idx"] = (
            (["None"] + sortable_columns).index(sort_col) if sort_col in (["None"] + sortable_columns) else 0
        )
    
    # Sort Order
    sort_order_key = f"{key_prefix}_sort_order"
    with col3:
        sort_order = st.selectbox(
            "⬆️⬇️ Order",
            ["Ascending ⬆️", "Descending ⬇️"],
            index=st.session_state.get(sort_order_key, 0),
            key=f"{sort_order_key}_select",
        )
        ascending = "Ascending" in sort_order
        st.session_state[sort_order_key] = 0 if ascending else 1
    
    # Status Filter
    status_filter = None
    industry_filter = None
    
    with col4:
        if show_status_filter and "Status" in df.columns:
            status_options = ["All Statuses"] + sorted(df["Status"].dropna().unique().tolist())
            status_key = f"{key_prefix}_status_filter"
            status_filter = st.selectbox(
                "🎯 Status Filter",
                status_options,
                index=st.session_state.get(status_key, 0),
                key=f"{status_key}_select",
            )
            st.session_state[status_key] = status_options.index(status_filter)
            if status_filter == "All Statuses":
                status_filter = None
    
    # Industry Filter (in second row if needed)
    col5, col6, col7, col8 = st.columns(4)
    if show_industry_filter and "Industry" in df.columns:
        with col5:
            industry_options = ["All Industries"] + sorted(df["Industry"].dropna().unique().tolist())
            industry_key = f"{key_prefix}_industry_filter"
            industry_filter = st.selectbox(
                "🏢 Industry Filter",
                industry_options,
                index=st.session_state.get(industry_key, 0),
                key=f"{industry_key}_select",
            )
            st.session_state[industry_key] = industry_options.index(industry_filter)
            if industry_filter == "All Industries":
                industry_filter = None
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # APPLY FILTERS & SORTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    filtered_df = df.copy()
    
    # Apply search across all columns
    if search_term:
        search_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        for col in filtered_df.columns:
            col_values = filtered_df[col].astype(str).str.lower()
            search_mask |= col_values.str.contains(search_term, na=False, regex=False)
        filtered_df = filtered_df[search_mask]
    
    # Apply status filter
    if status_filter:
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]
    
    # Apply industry filter
    if industry_filter:
        filtered_df = filtered_df[filtered_df["Industry"] == industry_filter]
    
    # Apply sorting
    if sort_col != "None":
        try:
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending, na_position='last')
        except Exception as e:
            st.warning(f"Could not sort by {sort_col}: {e}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DISPLAY TABLE WITH FORMATTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    st.markdown(f"**📋 Results: {len(filtered_df)} of {len(df)} rows**")
    
    # Format numeric columns for display
    display_df = filtered_df.copy()
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )
    
    # Display as dataframe (Streamlit's native table)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        key=f"{key_prefix}_dataframe",
    )
    
    return filtered_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    st.set_page_config(page_title="Interactive Table Demo", layout="wide")
    
    # Apply Bloomberg theme
    apply_bloomberg_theme()
    
    st.title("📊 Interactive Table Component Demo")
    st.markdown("### Bloomberg Terminal Theme - Dark Blue Sidebar + Beige Main Area")
    
    # Sample data
    sample_data = {
        "Symbol": ["NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCAP", "SMALLCAP", "NIFTY200", "NIFTYIT"],
        "Company Name": ["Nifty 50", "Bank Nifty", "Financial Nifty", "Nifty Midcap", "Nifty Smallcap", "Nifty 200", "Nifty IT"],
        "Status": ["Leading", "Improving", "Lagging", "Weakening", "Leading", "Improving", "Lagging"],
        "Industry": ["Index", "Financial", "Financial", "Index", "Index", "Index", "Technology"],
        "RS-Ratio": [102.5, 101.2, 98.3, 103.1, 104.5, 100.8, 97.2],
        "RS-Momentum": [103.2, 102.8, 96.5, 101.9, 105.3, 99.6, 95.1],
        "Price %Δ": [2.45, 1.82, -0.93, 3.21, 4.15, 1.67, -1.23],
    }
    
    df = pd.DataFrame(sample_data)
    
    st.markdown("### Features:")
    st.markdown("""
    ✅ **Search** - Type to search across all columns  
    ✅ **Sort** - Click dropdown to sort by any column  
    ✅ **Filter** - Use status and industry filters  
    ✅ **Responsive** - Adapts to all screen sizes  
    ✅ **Numeric Formatting** - Auto-formats decimal numbers  
    ✅ **Bloomberg Theme** - Dark blue sidebar + beige background
    """)
    
    filtered_df = render_interactive_table(
        df,
        key_prefix="demo_table",
        show_status_filter=True,
        show_industry_filter=True,
        sortable_columns=["Symbol", "Status", "RS-Ratio", "RS-Momentum", "Price %Δ"],
        numeric_columns=["RS-Ratio", "RS-Momentum", "Price %Δ"],
    )
    
    st.markdown("---")
    st.markdown("### Filtered Data (Downloadable):")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="rrg_indices.csv",
        mime="text/csv",
    )
