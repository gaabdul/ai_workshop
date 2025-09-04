import streamlit as st
import pandas as pd
import duckdb
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Chart rendering functions
def render_simple_timeseries(df, time_col, value_col, agg, show_labels):
    """Render simple timeseries chart"""
    try:
        if agg == "COUNT":
            agg_df = df.groupby(time_col).size().reset_index(name=value_col)
        else:
            agg_df = df.groupby(time_col)[value_col].agg(agg.lower()).reset_index()
        
        fig = px.line(agg_df, x=time_col, y=value_col, markers=True)
        fig.update_layout(title=f"{agg} of {value_col} over time")
        if show_labels:
            fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

def render_histogram(df, value_col, bins):
    """Render histogram chart"""
    try:
        fig = px.histogram(df, x=value_col, nbins=bins)
        fig.update_layout(title=f"Distribution of {value_col}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

def render_pie(df, category_col, value_col):
    """Render pie chart"""
    try:
        agg_df = df.groupby(category_col)[value_col].sum().reset_index()
        fig = px.pie(agg_df, values=value_col, names=category_col)
        fig.update_layout(title=f"{value_col} by {category_col}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

def render_boxplot(df, category_col, value_col):
    """Render boxplot chart"""
    try:
        fig = px.box(df, x=category_col, y=value_col)
        fig.update_layout(title=f"{value_col} distribution by {category_col}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

def render_heatmap(df, x_col, y_col, value_col):
    """Render heatmap chart"""
    try:
        pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
        fig = px.imshow(pivot_df, title=f"{value_col} heatmap")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

def render_advanced_timeseries(df, time_col, value_cols, agg):
    """Render advanced timeseries with multiple values"""
    try:
        fig = go.Figure()
        
        for col in value_cols:
            if agg == "COUNT":
                agg_df = df.groupby(time_col).size().reset_index(name=col)
            else:
                agg_df = df.groupby(time_col)[col].agg(agg.lower()).reset_index()
            
            fig.add_trace(go.Scatter(x=agg_df[time_col], y=agg_df[col], name=col, mode='lines+markers'))
        
        fig.update_layout(title=f"{agg} of multiple values over time", xaxis_title=time_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render chart: {str(e)}")

st.header("Analytics App Starter")

# Navigation
nav_selection = st.selectbox(
    "Navigation",
    ["Schema", "Data Explorer", "Analysis"],
    label_visibility="collapsed"
)

# Initialize DuckDB connection
db_path = "app.duckdb"
conn = duckdb.connect(db_path)

if nav_selection == "Schema":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Sanitize filename to snake_case table name
        filename = uploaded_file.name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename.replace('.csv', '')).lower()
        if not table_name or table_name.startswith('_'):
            table_name = "dataset"
        
        # Create/replace table in DuckDB
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        st.success(f"✅ Data saved to {db_path} | Table: {table_name} | Rows: {row_count}")
        
        st.dataframe(df)

    # Table selector and preview
    st.subheader("Database Tables")
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if tables:
            # Filter out internal tables
            table_names = [table[0] for table in tables if not table[0].startswith('_')]
            selected_table = st.selectbox("Select a table to preview:", table_names)
            
            if selected_table:
                preview_df = conn.execute(f"SELECT * FROM {selected_table} LIMIT 100").df()
                st.dataframe(preview_df)
                
                # Metrics Panel
                st.subheader("Metrics Panel")
                
                # Ensure _metrics table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS _metrics (
                        dataset VARCHAR,
                        alias VARCHAR,
                        display_name VARCHAR,
                        expression VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Metrics form
                with st.form("add_metric"):
                    metric_name = st.text_input("Display Name (required)", key="metric_name")
                    metric_expression = st.text_area("Expression (DuckDB SQL)", key="metric_expression", 
                                                   placeholder=f"SELECT column_name FROM {selected_table}")
                    
                    if st.form_submit_button("Save Metric"):
                        if metric_name and metric_expression:
                            try:
                                # Check if metric name already exists for this dataset
                                existing = conn.execute(
                                    "SELECT display_name FROM _metrics WHERE dataset = ? AND display_name = ?", 
                                    [selected_table, metric_name]
                                ).fetchone()
                                
                                if existing:
                                    st.error("Metric name already exists for this dataset")
                                else:
                                    # Generate unique alias
                                    alias = f"metric_{len(conn.execute('SELECT * FROM _metrics WHERE dataset = ?', [selected_table]).fetchall()) + 1}"
                                    
                                    # Test the expression
                                    test_query = f"SELECT {metric_expression} FROM {selected_table} LIMIT 1"
                                    conn.execute(test_query)
                                    
                                    # Save metric
                                    conn.execute(
                                        "INSERT INTO _metrics (dataset, alias, display_name, expression) VALUES (?, ?, ?, ?)",
                                        [selected_table, alias, metric_name, metric_expression]
                                    )
                                    st.success("Metric saved successfully!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Invalid expression: {str(e)}")
                        else:
                            st.error("Both fields are required")
                
                # Metrics list
                metrics = conn.execute(
                    "SELECT display_name, expression, created_at FROM _metrics WHERE dataset = ? ORDER BY created_at DESC",
                    [selected_table]
                ).fetchall()
                
                if metrics:
                    st.subheader("Metrics")
                    for i, (display_name, expression, created_at) in enumerate(metrics):
                        col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                        
                        with col1:
                            st.write(f"**{display_name}**")
                        with col2:
                            st.code(expression, language="sql")
                        with col3:
                            st.write(f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}")
                        with col4:
                            if st.button("Preview", key=f"preview_{i}"):
                                try:
                                    preview_query = f"SELECT *, ({expression}) as {display_name} FROM {selected_table} LIMIT 20"
                                    metric_preview = conn.execute(preview_query).df()
                                    st.dataframe(metric_preview)
                                except Exception as e:
                                    st.error(f"Preview error: {str(e)}")
                            
                            if st.button("Delete", key=f"delete_{i}"):
                                conn.execute(
                                    "DELETE FROM _metrics WHERE dataset = ? AND display_name = ?",
                                    [selected_table, display_name]
                                )
                                st.success("Metric deleted!")
                                st.rerun()
                else:
                    st.info("No metrics defined for this dataset")
                    
        else:
            st.info("No tables found in database")
    except Exception as e:
        st.info("No tables found in database")

elif nav_selection == "Data Explorer":
    st.subheader("Data Explorer")
    
    # Dataset selector
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if tables:
            # Filter out internal tables
            table_names = [table[0] for table in tables if not table[0].startswith('_')]
            selected_dataset = st.selectbox("Select a dataset:", table_names, key="explorer_dataset")
            
            if selected_dataset:
                # Robust schema read for base columns
                info_df = conn.execute(f"PRAGMA table_info('{selected_dataset}')").df()
                base_cols = info_df["name"].tolist()
                base_types = dict(zip(info_df["name"], info_df["type"]))
                
                # Type helpers
                def is_temporal(t):
                    return t.upper() in {"DATE", "TIMESTAMP", "TIMESTAMPTZ"}
                
                def is_numeric(t):
                    return t.upper() in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "REAL", "DOUBLE", "DECIMAL"}
                
                def is_text(t):
                    return t.upper() in {"VARCHAR", "CHAR", "TEXT"}
                
                # Filters UI
                with st.expander("Filters", expanded=True):
                    where_clauses = []
                    filter_params = []
                    
                    # Date filter
                    date_cols = [c for c in base_cols if is_temporal(base_types[c])]
                    if date_cols:
                        date_col = st.selectbox("Date column", date_cols, key="date_col")
                        
                        # Get min/max dates
                        date_range = conn.execute(f'''
                            SELECT MIN(CAST("{date_col}" AS DATE)) AS dmin, 
                                   MAX(CAST("{date_col}" AS DATE)) AS dmax 
                            FROM "{selected_dataset}"
                        ''').fetchone()
                        
                        if date_range[0] and date_range[1]:
                            d_start, d_end = st.date_input(
                                "Date range",
                                value=(date_range[0], date_range[1]),
                                key="date_range"
                            )
                            
                            if d_start and d_end:
                                where_clauses.append(f'CAST(b."{date_col}" AS DATE) BETWEEN ? AND ?')
                                filter_params.extend([d_start, d_end])
                    
                    # Numeric filters (first 6)
                    numeric_cols = [c for c in base_cols if is_numeric(base_types[c])][:6]
                    for col in numeric_cols:
                        try:
                            num_range = conn.execute(f'''
                                SELECT MIN("{col}") AS vmin, MAX("{col}") AS vmax 
                                FROM "{selected_dataset}"
                            ''').fetchone()
                            
                            if num_range[0] is not None and num_range[1] is not None:
                                if base_types[col].upper() in {"REAL", "DOUBLE", "DECIMAL"}:
                                    min_val, max_val = st.slider(
                                        f"{col} range",
                                        float(num_range[0]), float(num_range[1]),
                                        (float(num_range[0]), float(num_range[1])),
                                        key=f"num_{col}"
                                    )
                                else:
                                    min_val, max_val = st.slider(
                                        f"{col} range",
                                        int(num_range[0]), int(num_range[1]),
                                        (int(num_range[0]), int(num_range[1])),
                                        key=f"num_{col}"
                                    )
                                
                                if min_val != num_range[0] or max_val != num_range[1]:
                                    where_clauses.append(f'b."{col}" BETWEEN ? AND ?')
                                    filter_params.extend([min_val, max_val])
                        except:
                            continue
                    
                    # Categorical filters (low-cardinality)
                    text_cols = [c for c in base_cols if is_text(base_types[c])]
                    for col in text_cols:
                        try:
                            distinct_count = conn.execute(f'''
                                SELECT COUNT(DISTINCT "{col}") AS n FROM "{selected_dataset}"
                            ''').fetchone()[0]
                            
                            if distinct_count <= 50:
                                distinct_values = conn.execute(f'''
                                    SELECT DISTINCT "{col}" AS v FROM "{selected_dataset}" 
                                    ORDER BY 1 LIMIT 50
                                ''').fetchall()
                                
                                if distinct_values:
                                    values_list = [v[0] for v in distinct_values if v[0] is not None]
                                    selected_values = st.multiselect(
                                        f"{col} (select values)",
                                        values_list,
                                        default=values_list,
                                        key=f"cat_{col}"
                                    )
                                    
                                    if len(selected_values) != len(values_list):
                                        placeholders = ', '.join(['?' for _ in selected_values])
                                        where_clauses.append(f'b."{col}" IN ({placeholders})')
                                        filter_params.extend(selected_values)
                        except:
                            continue
                
                # Ensure _views table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS _views(
                        dataset TEXT NOT NULL,
                        name    TEXT NOT NULL,
                        config_json JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT now(),
                        UNIQUE(dataset, name)
                    )
                """)
                
                # Load metrics for the dataset
                mdf = conn.execute("""
                    SELECT display_name, expression
                    FROM _metrics
                    WHERE dataset = ?
                    ORDER BY created_at DESC
                """, [selected_dataset]).df() if selected_dataset else None
                
                metric_cols = [] if mdf is None else mdf["display_name"].tolist()
                
                # Assemble preview query with filters
                if not metric_cols:
                    # No metrics, simple query
                    if where_clauses:
                        where_clause = " WHERE " + " AND ".join(where_clauses)
                        preview_query = f'SELECT * FROM "{selected_dataset}" b{where_clause} LIMIT 200'
                        preview_df = conn.execute(preview_query, filter_params).df()
                    else:
                        preview_query = f'SELECT * FROM "{selected_dataset}" LIMIT 200'
                        preview_df = conn.execute(preview_query).df()
                else:
                    # Build query with metrics
                    metric_expressions = []
                    for _, row in mdf.iterrows():
                        safe_alias = re.sub(r'[^0-9A-Za-z_]+', '_', row['display_name'])
                        metric_expressions.append(f'{row["expression"]} AS "{safe_alias}"')
                    
                    metric_select = ', '.join(metric_expressions)
                    
                    if where_clauses:
                        where_clause = " WHERE " + " AND ".join(where_clauses)
                        preview_query = f'''
                            SELECT b.*, {metric_select}
                            FROM "{selected_dataset}" b{where_clause}
                            LIMIT 200
                        '''
                        preview_df = conn.execute(preview_query, filter_params).df()
                    else:
                        preview_query = f'''
                            SELECT b.*, {metric_select}
                            FROM "{selected_dataset}" b
                            LIMIT 200
                        '''
                        preview_df = conn.execute(preview_query).df()
                
                # Visible columns multiselect
                all_cols = base_cols + metric_cols
                visible_cols = st.multiselect(
                    "Visible columns:",
                    all_cols,
                    default=all_cols,
                    key="visible_cols"
                )
                
                # Show metric columns hint
                if metric_cols:
                    st.markdown(f"""
                    <div style="background-color: #e6f3ff; padding: 8px; border-radius: 4px; margin: 10px 0;">
                        <strong>Calculated columns:</strong> {', '.join(metric_cols)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Saved Views UI
                st.subheader("Saved Views")
                
                # Load existing views
                vdf = conn.execute("SELECT name, config_json FROM _views WHERE dataset = ? ORDER BY created_at DESC", [selected_dataset]).df()
                existing_views = vdf["name"].tolist() if not vdf.empty else []
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_view = st.selectbox("Load saved view", ["None"] + existing_views, key="load_view")
                    
                    if selected_view != "None":
                        if st.button("Apply View", key="apply_view"):
                            try:
                                import json
                                view_config = json.loads(vdf[vdf["name"] == selected_view]["config_json"].iloc[0])
                                
                                # Apply visible columns
                                st.session_state.visible_cols = view_config["visible_cols"]
                                
                                # Apply date filter
                                if view_config["filters"]["date"]["col"]:
                                    st.session_state.date_col = view_config["filters"]["date"]["col"]
                                    if view_config["filters"]["date"]["start"] and view_config["filters"]["date"]["end"]:
                                        st.session_state.date_range = (
                                            view_config["filters"]["date"]["start"], 
                                            view_config["filters"]["date"]["end"]
                                        )
                                
                                # Apply numeric filters
                                for col, (min_val, max_val) in view_config["filters"]["numeric"].items():
                                    st.session_state[f"num_{col}"] = (min_val, max_val)
                                
                                # Apply categorical filters
                                for col, values in view_config["filters"]["categorical"].items():
                                    st.session_state[f"cat_{col}"] = values
                                
                                st.success(f"Applied view '{selected_view}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error applying view: {str(e)}")
                
                with col2:
                    if selected_view != "None":
                        if st.button("Delete View", key="delete_view"):
                            conn.execute("DELETE FROM _views WHERE dataset = ? AND name = ?", [selected_dataset, selected_view])
                            st.success(f"Deleted view '{selected_view}'")
                            st.rerun()
                
                # Save current view
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    view_name = st.text_input("View name", key="save_view_name")
                
                with col2:
                    overwrite = st.checkbox("Overwrite if exists", key="overwrite_view")
                
                with col3:
                    if st.button("Save View", disabled=not view_name, key="save_view"):
                        try:
                            import json
                            
                            # Build current config
                            numeric_ranges = {}
                            for col in numeric_cols:
                                if f"num_{col}" in st.session_state:
                                    numeric_ranges[col] = st.session_state[f"num_{col}"]
                            
                            categorical_values = {}
                            for col in text_cols:
                                if f"cat_{col}" in st.session_state:
                                    categorical_values[col] = st.session_state[f"cat_{col}"]
                            
                            view_cfg = {
                                "dataset": selected_dataset,
                                "visible_cols": visible_cols,
                                "filters": {
                                    "date": {
                                        "col": st.session_state.get("date_col", None),
                                        "start": str(st.session_state.get("date_range", (None, None))[0]) if st.session_state.get("date_range") else None,
                                        "end": str(st.session_state.get("date_range", (None, None))[1]) if st.session_state.get("date_range") else None
                                    },
                                    "numeric": numeric_ranges,
                                    "categorical": categorical_values
                                }
                            }
                            
                            payload = json.dumps(view_cfg)
                            
                            if overwrite:
                                conn.execute("DELETE FROM _views WHERE dataset = ? AND name = ?", [selected_dataset, view_name])
                            
                            conn.execute("INSERT INTO _views(dataset,name,config_json) VALUES (?,?,?)", [selected_dataset, view_name, payload])
                            st.success(f"Saved view '{view_name}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving view: {str(e)}")
                
                # Show table with selected columns
                if visible_cols and len(preview_df) > 0:
                    filtered_df = preview_df[visible_cols]
                    st.dataframe(filtered_df)
                elif len(preview_df) > 0:
                    st.info("Select columns to display")
                else:
                    st.info("No data to display")
            else:
                st.info("Select a dataset above to explore data")
        else:
            st.info("No datasets found in database")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

elif nav_selection == "Analysis":
    st.subheader("Analysis")
    
    # Dataset selector
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if tables:
            # Filter out internal tables
            table_names = [table[0] for table in tables if not table[0].startswith('_')]
            selected_dataset = st.selectbox("Select a dataset:", table_names, key="analysis_dataset")
            
            if selected_dataset:
                # Chart type selector
                chart_types = [
                    "Simple Timeseries",
                    "Histogram", 
                    "Pie",
                    "Boxplot",
                    "Heatmap",
                    "Advanced Timeseries"
                ]
                chart_type = st.selectbox("Select chart type:", chart_types, key="chart_type")
                
                # Get dataset info
                info_df = conn.execute(f"PRAGMA table_info('{selected_dataset}')").df()
                base_cols = info_df["name"].tolist()
                base_types = dict(zip(info_df["name"], info_df["type"]))
                
                # Load metrics for the dataset
                mdf = conn.execute("""
                    SELECT display_name, expression
                    FROM _metrics
                    WHERE dataset = ?
                    ORDER BY created_at DESC
                """, [selected_dataset]).df() if selected_dataset else None
                
                metric_cols = [] if mdf is None else mdf["display_name"].tolist()
                all_cols = base_cols + metric_cols
                
                # Load data with TRY_CAST for time and numeric columns
                if chart_type == "Simple Timeseries":
                    col1, col2 = st.columns(2)
                    with col1:
                        time_col = st.selectbox("Time column:", all_cols, key="ts_time_col")
                    with col2:
                        value_col = st.selectbox("Value column:", all_cols, key="ts_value_col")
                    
                    agg = st.selectbox("Aggregation:", ["SUM", "AVG", "COUNT"], key="ts_agg")
                    show_labels = st.checkbox("Show value labels", key="ts_labels")
                    
                    if st.button("Generate Chart", key="ts_generate"):
                        try:
                            # Build query with TRY_CAST
                            if agg == "COUNT":
                                query = f"""
                                    SELECT TRY_CAST("{time_col}" AS DATE) as time_col, 
                                           COUNT(*) as value_col
                                    FROM "{selected_dataset}"
                                    WHERE TRY_CAST("{time_col}" AS DATE) IS NOT NULL
                                    GROUP BY TRY_CAST("{time_col}" AS DATE)
                                    ORDER BY time_col
                                """
                            else:
                                query = f"""
                                    SELECT TRY_CAST("{time_col}" AS DATE) as time_col, 
                                           {agg}(TRY_CAST("{value_col}" AS DOUBLE)) as value_col
                                    FROM "{selected_dataset}"
                                    WHERE TRY_CAST("{time_col}" AS DATE) IS NOT NULL
                                    GROUP BY TRY_CAST("{time_col}" AS DATE)
                                    ORDER BY time_col
                                """
                            
                            df = conn.execute(query).df()
                            if not df.empty:
                                render_simple_timeseries(df, "time_col", "value_col", agg, show_labels)
                            else:
                                st.info("No data available for the selected columns")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
                
                elif chart_type == "Histogram":
                    value_col = st.selectbox("Value column:", all_cols, key="hist_value_col")
                    bins = st.slider("Number of bins:", 5, 50, 20, key="hist_bins")
                    
                    if st.button("Generate Chart", key="hist_generate"):
                        try:
                            query = f"""
                                SELECT TRY_CAST("{value_col}" AS DOUBLE) as value_col
                                FROM "{selected_dataset}"
                                WHERE TRY_CAST("{value_col}" AS DOUBLE) IS NOT NULL
                            """
                            df = conn.execute(query).df()
                            if not df.empty:
                                render_histogram(df, "value_col", bins)
                            else:
                                st.info("No numeric data available for the selected column")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
                
                elif chart_type == "Pie":
                    col1, col2 = st.columns(2)
                    with col1:
                        category_col = st.selectbox("Category column:", base_cols, key="pie_cat_col")
                    with col2:
                        value_col = st.selectbox("Value column:", all_cols, key="pie_value_col")
                    
                    if st.button("Generate Chart", key="pie_generate"):
                        try:
                            query = f"""
                                SELECT "{category_col}" as category_col,
                                       SUM(TRY_CAST("{value_col}" AS DOUBLE)) as value_col
                                FROM "{selected_dataset}"
                                WHERE TRY_CAST("{value_col}" AS DOUBLE) IS NOT NULL
                                GROUP BY "{category_col}"
                            """
                            df = conn.execute(query).df()
                            if not df.empty:
                                render_pie(df, "category_col", "value_col")
                            else:
                                st.info("No data available for the selected columns")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
                
                elif chart_type == "Boxplot":
                    col1, col2 = st.columns(2)
                    with col1:
                        category_col = st.selectbox("Category column:", base_cols, key="box_cat_col")
                    with col2:
                        value_col = st.selectbox("Value column:", all_cols, key="box_value_col")
                    
                    if st.button("Generate Chart", key="box_generate"):
                        try:
                            query = f"""
                                SELECT "{category_col}" as category_col,
                                       TRY_CAST("{value_col}" AS DOUBLE) as value_col
                                FROM "{selected_dataset}"
                                WHERE TRY_CAST("{value_col}" AS DOUBLE) IS NOT NULL
                            """
                            df = conn.execute(query).df()
                            if not df.empty:
                                render_boxplot(df, "category_col", "value_col")
                            else:
                                st.info("No numeric data available for the selected columns")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
                
                elif chart_type == "Heatmap":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_col = st.selectbox("X column:", base_cols, key="heat_x_col")
                    with col2:
                        y_col = st.selectbox("Y column:", base_cols, key="heat_y_col")
                    with col3:
                        value_col = st.selectbox("Value column:", all_cols, key="heat_value_col")
                    
                    if st.button("Generate Chart", key="heat_generate"):
                        try:
                            query = f"""
                                SELECT "{x_col}" as x_col,
                                       "{y_col}" as y_col,
                                       TRY_CAST("{value_col}" AS DOUBLE) as value_col
                                FROM "{selected_dataset}"
                                WHERE TRY_CAST("{value_col}" AS DOUBLE) IS NOT NULL
                            """
                            df = conn.execute(query).df()
                            if not df.empty:
                                render_heatmap(df, "x_col", "y_col", "value_col")
                            else:
                                st.info("No numeric data available for the selected columns")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
                
                elif chart_type == "Advanced Timeseries":
                    time_col = st.selectbox("Time column:", all_cols, key="adv_ts_time_col")
                    value_cols = st.multiselect("Value columns:", all_cols, key="adv_ts_value_cols")
                    agg = st.selectbox("Aggregation:", ["SUM", "AVG", "COUNT"], key="adv_ts_agg")
                    
                    if st.button("Generate Chart", key="adv_ts_generate"):
                        try:
                            if not value_cols:
                                st.info("Please select at least one value column")
                            else:
                                # Build query for each value column
                                all_data = []
                                for col in value_cols:
                                    if agg == "COUNT":
                                        query = f"""
                                            SELECT TRY_CAST("{time_col}" AS DATE) as time_col,
                                                   COUNT(*) as value_col,
                                                   '{col}' as column_name
                                            FROM "{selected_dataset}"
                                            WHERE TRY_CAST("{time_col}" AS DATE) IS NOT NULL
                                            GROUP BY TRY_CAST("{time_col}" AS DATE)
                                        """
                                    else:
                                        query = f"""
                                            SELECT TRY_CAST("{time_col}" AS DATE) as time_col,
                                                   {agg}(TRY_CAST("{col}" AS DOUBLE)) as value_col,
                                                   '{col}' as column_name
                                            FROM "{selected_dataset}"
                                            WHERE TRY_CAST("{col}" AS DOUBLE) IS NOT NULL
                                            GROUP BY TRY_CAST("{time_col}" AS DATE)
                                        """
                                    
                                    col_df = conn.execute(query).df()
                                    if not col_df.empty:
                                        all_data.append(col_df)
                                
                                if all_data:
                                    # Combine all data
                                    combined_df = pd.concat(all_data, ignore_index=True)
                                    render_advanced_timeseries(combined_df, "time_col", "column_name", agg)
                                else:
                                    st.info("No data available for the selected columns")
                        except Exception as e:
                            st.info(f"Could not generate chart: {str(e)}")
            else:
                st.info("Select a dataset above to create charts")
        else:
            st.info("No datasets found in database")
    except Exception as e:
        st.error(f"Error loading analysis: {str(e)}")

conn.close()
