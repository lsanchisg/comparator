import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page to wide mode
st.set_page_config(layout="wide")
st.title("🔬 2D Colormap Comparison Tool")
st.write("For Mesh Convergence Analysis")

# --- 1. Data Loading and Caching ---
@st.cache_data
def load_data(file1, file2):
    """
    Loads, merges, and processes data from two CSV files.
    """
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure the CSV files are in the same directory as the app.")
        return None, [], None, None

    # Clean column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Get available parameters (all columns except the axes)
    params = [col for col in df1.columns if col not in ['h_fib', 'lda0']]
    
    # Merge dataframes
    df_merged = pd.merge(df1, df2, on=['h_fib', 'lda0'], suffixes=('_1', '_2'))
    
    # Calculate difference columns for all parameters
    for p in params:
        col_1 = f"{p}_1"
        col_2 = f"{p}_2"
        col_diff = f"{p}_Diff"
        if col_1 in df_merged and col_2 in df_merged:
            df_merged[col_diff] = (df_merged[col_1] - df_merged[col_2]).abs()
    
    # Get axis values
    x_vals = sorted(df_merged['lda0'].unique())
    y_vals = sorted(df_merged['h_fib'].unique())
    
    return df_merged, params, x_vals, y_vals

# Load data using the files you provided
FILE_1 = 'EXPORT_tbl2_0_05.csv'
FILE_2 = 'EXPORT_tbl2_0_07.csv'
df, params, x_vals, y_vals = load_data(FILE_1, FILE_2)

if df is not None:
    st.sidebar.info(f"**File 1:** `{FILE_1}`\n\n**File 2:** `{FILE_2}`")

    # --- 2. Sidebar Controls ---
    selected_param = st.sidebar.selectbox("Select Parameter to Plot:", params)
    
    display_mode = st.sidebar.radio(
        "Select Heatmap Display:",
        ("File 1 (0.05)", "File 2 (0.07)", "Absolute Difference"),
        index=2  # Default to "Absolute Difference"
    )
    
    st.sidebar.markdown("---")
    
    # --- 3. Sliders for Cross-Sections ---
    y_slider_val = st.sidebar.slider(
        "Fiber Height (h_fib) Cross-Section",
        min_value=min(y_vals),
        max_value=max(y_vals),
        value=y_vals[len(y_vals) // 2],
        step=(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 0.0
    )
    
    x_slider_val = st.sidebar.slider(
        "Wavelength (lda0) Cross-Section",
        min_value=min(x_vals),
        max_value=max(x_vals),
        value=x_vals[len(x_vals) // 2],
        step=(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.0
    )

    # Find the closest index in our data for the slider values
    y_slider_idx = (np.abs(np.array(y_vals) - y_slider_val)).argmin()
    x_slider_idx = (np.abs(np.array(x_vals) - x_slider_val)).argmin()

    # --- 4. Data Pivoting ---
    # Create 2D grids (numpy arrays) for each dataset
    def pivot_data(df, value_col):
        try:
            return df.pivot(index='h_fib', columns='lda0', values=value_col).values
        except Exception as e:
            st.error(f"Error pivoting data for {value_col}: {e}")
            return np.zeros((len(y_vals), len(x_vals)))

    z_data_1 = pivot_data(df, f"{selected_param}_1")
    z_data_2 = pivot_data(df, f"{selected_param}_2")
    z_data_diff = pivot_data(df, f"{selected_param}_Diff")

    # --- 5. Plotting ---
    
    # Create the 3-plot layout
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        specs=[
            [{"type": "xy"}, {}],  # Top plot (row 1, col 1)
            [{"type": "heatmap"}, {"type": "xy"}]  # Main (2,1), Right (2,2)
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )

    # --- Plot 1: Top Cross-Section (vs. lda0) ---
    # This plot always compares the two original files
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=z_data_1[y_slider_idx, :], # Row from slider
        name='File 1 (0.05)',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=z_data_2[y_slider_idx, :], # Row from slider
        name='File 2 (0.07)',
        line=dict(color='red', dash='dot')
    ), row=1, col=1)

    # --- Plot 2: Main Heatmap ---
    if display_mode == "File 1 (0.05)":
        z_data = z_data_1
        colorscale = 'Viridis'
        heatmap_title = f"{selected_param} (File 1: 0.05)"
    elif display_mode == "File 2 (0.07)":
        z_data = z_data_2
        colorscale = 'Viridis'
        heatmap_title = f"{selected_param} (File 2: 0.07)"
    else: # "Absolute Difference"
        z_data = z_data_diff
        colorscale = 'Reds' # 'Reds' or 'Hot' is good for 0-max values
        heatmap_title = f"Absolute Difference in {selected_param}"

    fig.add_trace(go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_data,
        colorscale=colorscale,
        colorbar_title=selected_param,
        zmin=0 if display_mode == "Absolute Difference" else None, # Force diff to start at 0
        zmax=z_data_diff.max() if display_mode == "Absolute Difference" else None
    ), row=2, col=1)
    
    # Add slider lines to heatmap
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=[y_slider_val] * len(x_vals),
        mode='lines',
        line=dict(color='white', dash='dash', width=1),
        name='h_fib cut'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_slider_val] * len(y_vals),
        y=y_vals,
        mode='lines',
        line=dict(color='white', dash='dash', width=1),
        name='lda0 cut'
    ), row=2, col=1)

    # --- Plot 3: Right Cross-Section (vs. h_fib) ---
    # This plot also always compares the two original files
    fig.add_trace(go.Scatter(
        y=y_vals, # Y-axis is h_fib
        x=z_data_1[:, x_slider_idx], # X-axis is value (column from slider)
        name='File 1 (0.05)',
        line=dict(color='blue'),
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        y=y_vals, # Y-axis is h_fib
        x=z_data_2[:, x_slider_idx], # X-axis is value (column from slider)
        name='File 2 (0.07)',
        line=dict(color='red', dash='dot'),
        showlegend=False
    ), row=2, col=2)

    # --- 6. Layout Updates ---
    fig.update_layout(
        height=700,
        title_text=heatmap_title,
        
        # Top plot
        xaxis1=dict(showticklabels=False),
        yaxis1=dict(title=selected_param),
        
        # Main plot
        xaxis2=dict(title='lda0 (Wavelength)'),
        yaxis2=dict(title='h_fib (Fiber Height)'),
        
        # Right plot
        xaxis3=dict(title=selected_param),
        yaxis3=dict(showticklabels=False),
        
        # Link axes
        xaxis1_matches='x2',
        yaxis3_matches='y2',
        
        # Legend
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)