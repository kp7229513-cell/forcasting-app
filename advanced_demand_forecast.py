import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="AI Demand Forecaster Pro", layout="wide")

st.title("🚀 Advanced Product Demand Forecasting AI")
st.markdown("""
यह मॉडल न केवल भविष्य की सेल्स बताता है, बल्कि **Holidays** और **Seasonality** (हफ़्तों और महीनों का असर) को भी समझता है।
""")

# Sidebar for Settings
st.sidebar.header("Settings ⚙️")
days_to_predict = st.sidebar.slider("कितने दिनों का Forecast चाहिए?", 7, 365, 30)
include_holidays = st.sidebar.checkbox("Indian Holidays का असर शामिल करें?", value=True)

uploaded_file = st.file_uploader("अपनी Sales CSV फाइल अपलोड करें (Columns: Date, Sales)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Cleaning
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Raw Data (Last 5 Days)")
        st.write(df.tail())
    
    with col2:
        st.subheader("📊 Sales Trend")
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='ds', y='y', ax=ax)
        st.pyplot(fig)

    if st.button("Run Advanced Prediction"):
        with st.spinner('AI मॉडल डेटा को समझ रहा है... कृपया रुकें।'):
            
            # 1. Advanced Model Initialization
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            
            if include_holidays:
                model.add_country_holidays(country_name='IN') # India ke festivals add kiye
            
            model.fit(df)

            # 2. Future Prediction
            future = model.make_future_dataframe(periods=days_to_predict)
            forecast = model.predict(future)

            # 3. Visualizing Components (Trend vs Holidays)
            st.success(f"अगले {days_to_predict} दिनों का सफल विश्लेषण!")
            
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Graph", "🔍 Trend Analysis", "📦 Download Data"])
            
            with tab1:
                st.subheader("Future Demand Projection")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)
            
            with tab2:
                st.subheader("Market Trends (महीनों और हफ्तों का असर)")
                # Ye dikhayega ki Sunday ko sale badhti hai ya Diwali pe
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
            with tab3:
                # Prediction result download karne ke liye
                output_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict)
                output_df.columns = ['Date', 'Predicted_Demand', 'Min_Demand', 'Max_Demand']
                st.write(output_df)
                
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", data=csv, file_name='forecast_results.csv', mime='text/csv')

            # 4. Accuracy Calculation (Impressive Part for Teachers)
            try:
                df_cv = cross_validation(model, initial='365 days', period='30 days', horizon='30 days')
                df_p = performance_metrics(df_cv)
                accuracy = 100 - (df_p['mape'].mean() * 100)
                st.sidebar.metric("Model Accuracy (MAPE)", f"{accuracy:.2f}%")
            except:
                st.sidebar.warning("Accuracy calculate karne ke liye kam se kam 1 saal ka data chahiye.")

else:
    st.info("शुरू करने के लिए कृपया एक CSV फाइल अपलोड करें जिसमें 'Date' और 'Sales' कॉलम हों।")