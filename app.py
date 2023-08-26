import streamlit as st
import datetime
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

today = datetime.date.today()
today_eu = date.today().strftime("%d/%m/%Y")
start = today - datetime.timedelta(days=365.25 * 5)
end = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')
#st.text(today, align="left", fontsize=8)

# Apply CSS styling to move the text to the left upper corner


# Add text with today's date using st.markdown and the custom class
st.markdown('<div class="upper-left">Last update ' + str(today_eu) + '</div>', unsafe_allow_html=True)



stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
slected_stock = st.selectbox('Select Stocks', stocks)

n_years = st.slider('Years of prediction', 1, 4)
period = n_years*365

@st.cache_data
def load_data(stock):
    data  = yf.download(stock,start,end)
    data.reset_index(inplace=True)
    return data

data = load_data(slected_stock)

st.subheader('Stock data')
st.write(data.tail())

#plot
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock open'))
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock close'))
fig1.layout.update(title_text='Time series data',xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

#predict
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date':'ds','Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#plot forecast
st.subheader('Forecast data')
fig2 = plot_plotly(m,forecast)
st.plotly_chart(fig2)

@st.cache_data
def last_day()
    today = datetime.date.today()
    today_eu = date.today().strftime("%d/%m/%Y")
    return today_eu

lastday = last_day()
st.markdown('<div class="upper-left">Last update ' + str(lastday) + '</div>', unsafe_allow_html=True)

