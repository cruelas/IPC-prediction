from numpy import select
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Bola de Cristal - IPC")

stocks = ("NAFTRAC.MX", "AMXL.MX", "WALMEX.MX", "GFNORTEO.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "CEMEXCPO.MX", "TLEVISACPO.MX", "ELEKTRA.MX", "GAPB.MX", "ASURB.MX", "BIMBOA.MX", "KOFUBL.MX", "ORBIA.MX", "AC.MX", "KIMBERA.MX", "GRUMAB.MX", "ALFAA.MX", "GFINBURO.MX", "PE&OLES.MX", "OMAB.MX", "PINFRA.MX", "CUERVO.MX", "GCARSOA1.MX", "GCC.MX", "VESTA.MX", "BBAJIOO.MX", "MEGACPO.MX", "ALSEA.MX", "BOLSAA.MX", "LIVEPOLC-1.MX", "Q.MX", "SITESB-1.MX", "LABB.MX", "RA.MX")
selected_stock = st.selectbox("Selecciona la acción o el ETF que quieres que analice:", stocks)

n_years = st.slider("Meses a predecir:", 1, 12)
period = n_years*30

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Cargando datos...")
data = load_data(selected_stock)
data_load_state.text("Cargando datos... listo!")

st.subheader("Datos Observados")

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Adj Close"], name="Adj Close"))
    fig.layout.update(title_text=f"Precios Ajustados de Cierre - {selected_stock}", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = data[["Date", "Adj Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Predicción")

end_prediction = forecast.loc[forecast.index[-1], "yhat"]
end_observation = data.loc[data.index[-1], "Adj Close"]
pct_change = end_prediction/end_observation - 1

if pct_change < 0:
    st.write(f"El modelo prevé una pérdida de {100*pct_change:.2f}% para {selected_stock} dentro de {n_years} mes(es).")
else:
    st.write(f"El modelo prevé una ganancia de {100*pct_change:.2f}% para {selected_stock} dentro de {n_years} mes(es).")

def plot_forecast_data():
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)

def plot_forecast_data2():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Adj Close"], name="Datos observados"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicción del modelo"))
    fig.layout.update(title_text=f"Predicción de Precios Ajustados de Cierre - {selected_stock}", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_data2()

st.subheader("¿Cómo funciona?")
st.write("Bola de Cristal es un proyecto de predicción de series de tiempo para acciones del S&P/BMV IPC utilizando Prophet. Se utiliza un modelo aditivo que intenta encontrar tendencias no lineales y toma en cuenta efectos de estacionalidad en los datos.")

st.write(f"Para {selected_stock}, sus componentes aditivos son:")
fig_comp = m.plot_components(forecast)
st.write(fig_comp)