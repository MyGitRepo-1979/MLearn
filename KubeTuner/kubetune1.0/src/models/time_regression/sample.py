import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

def run_forecast_and_export():
    output_file_path = Path(__file__).resolve().parent / 'output' / 'kubetune_gb_predictions.xlsx'
    if not output_file_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_file_path}")

    df = pd.read_excel(output_file_path)
    if 'collectionTimestamp' in df.columns:
        df['collectionTimestamp'] = pd.to_datetime(df['collectionTimestamp'])
    else:
        raise KeyError("Column 'collectionTimestamp' not found in the data.")

    forecast_steps = 5
    results = []
    plot_info = []

    for deployment in df['deployment'].dropna().unique():
        dep_df = df[df['deployment'] == deployment].sort_values('collectionTimestamp')
        last_row = dep_df.iloc[-1]
        # --- Memory Forecast ---
        mem_series = dep_df['gb_predicted_memrequest']
        if len(mem_series) > 3:
            mem_model = ARIMA(mem_series, order=(1, 1, 1))
            mem_fit = mem_model.fit()
            mem_forecast = mem_fit.forecast(steps=forecast_steps)
            last_timestamp = dep_df['collectionTimestamp'].iloc[-1]
            freq = (dep_df['collectionTimestamp'].iloc[-1] - dep_df['collectionTimestamp'].iloc[-2]) if len(dep_df) > 1 else pd.Timedelta(days=1)
            mem_forecast_index = [last_timestamp + freq * (i + 1) for i in range(forecast_steps)]
            for ts, val in zip(mem_forecast_index, mem_forecast):
                results.append({
                    'deployment': deployment,
                    'resource': 'memory',
                    'forecast_timestamp': ts,
                    'forecasted_value': val,
                    'last_usage': last_row.get('memUsageMB', np.nan),
                    'last_request': last_row.get('memRequestMB', np.nan),
                    'last_predicted': last_row.get('gb_predicted_memrequest', np.nan),
                    'last_actual_timestamp': last_row.get('collectionTimestamp', pd.NaT)
                })
            plot_info.append((deployment, 'memory', dep_df['collectionTimestamp'], mem_series, mem_forecast_index, mem_forecast))
        # --- CPU Forecast ---
        cpu_series = dep_df['gb_predicted_cpurequest'] if 'gb_predicted_cpurequest' in dep_df.columns else None
        if cpu_series is not None and len(cpu_series) > 3:
            cpu_model = ARIMA(cpu_series, order=(1, 1, 1))
            cpu_fit = cpu_model.fit()
            cpu_forecast = cpu_fit.forecast(steps=forecast_steps)
            cpu_forecast_index = [last_timestamp + freq * (i + 1) for i in range(forecast_steps)]
            for ts, val in zip(cpu_forecast_index, cpu_forecast):
                results.append({
                    'deployment': deployment,
                    'resource': 'cpu',
                    'forecast_timestamp': ts,
                    'forecasted_value': val,
                    'last_usage': last_row.get('cpuUsage', np.nan),
                    'last_request': last_row.get('cpuRequest', np.nan),
                    'last_predicted': last_row.get('gb_predicted_cpurequest', np.nan),
                    'last_actual_timestamp': last_row.get('collectionTimestamp', pd.NaT)
                })
            plot_info.append((deployment, 'cpu', dep_df['collectionTimestamp'], cpu_series, cpu_forecast_index, cpu_forecast))
        else:
            print(f"Not enough data for deployment {deployment} to forecast CPU.")

    # Export all forecasts to Excel
    if results:
        forecast_df = pd.DataFrame(results)
        # Make all datetime columns timezone-unaware, even if not detected as datetimetz
        for col in forecast_df.select_dtypes(include=["datetimetz"]).columns:
            forecast_df[col] = forecast_df[col].dt.tz_localize(None)
        for col in forecast_df.select_dtypes(include=["datetime64[ns]"]).columns:
            forecast_df[col] = pd.to_datetime(forecast_df[col])
        export_path = Path(__file__).resolve().parent / 'output' / 'arima_forecast_results.xlsx'
        forecast_df.to_excel(export_path, index=False)
        print(f"ARIMA forecast results exported to {export_path}")

    return plot_info

def run_streamlit_app(plot_info):
    import streamlit as st
    import plotly.graph_objs as go

    st.title("ARIMA Forecast Visualizations by Deployment")
    deployments = sorted(set([x[0] for x in plot_info]))
    selected_deployment = st.selectbox("Select Deployment", deployments)

    for dep, resource, timestamps, series, forecast_index, forecast in plot_info:
        if dep == selected_deployment:
            st.subheader(f"{resource.capitalize()} Forecast for {dep}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, y=series,
                mode='lines+markers',
                name='Historical Predicted',
                hovertemplate='Time: %{x}<br>Value: %{y}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_index, y=forecast,
                mode='lines+markers',
                name='ARIMA Forecast',
                line=dict(dash='dash', color='red'),
                hovertemplate='Forecast Time: %{x}<br>Forecast: %{y}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title=f"Predicted {resource.capitalize()} Request",
                legend_title="Legend",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    plot_info = run_forecast_and_export()
    # To launch the Streamlit app, run this file with: streamlit run sample.py
    try:
        import streamlit as st
        run_streamlit_app(plot_info)
    except ImportError:
        print("Streamlit is not installed. To use the interactive visualization, install streamlit and run:\nstreamlit run sample.py")
