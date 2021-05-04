import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import calendar
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from PIL import Image

url1 = 'https://drive.google.com/file/d/1XMaIVbV2l1WUKNQPvP7J1X1sbT1zRtY8/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
df_train=pd.read_csv(path1)

url2 = 'https://drive.google.com/file/d/1QcnGtZKHdhTjE-oBjvVeRh22aNdO35Nk/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
df_store=pd.read_csv(path2)

df_train['Year'] = pd.DatetimeIndex(df_train['Date']). year
df_train['Month'] = pd.DatetimeIndex(df_train['Date']). month
df_train['Quarter']=pd.DatetimeIndex(df_train['Date']).quarter
df_train=df_train[df_train['Year']!=2015]
df_train=df_train.join(df_store.set_index('Store'),on='Store')
df_train['Store']=df_train['Store'].astype(str)
df_train['Quarter']=df_train['Quarter'].astype(str)
df_train['Month'] = df_train['Month'].apply(lambda x: calendar.month_abbr[x])

#converting DayOfWeek from number to day name
df_train['DayOfWeek'] = df_train['DayOfWeek'] -1 #days should be from 0 to 6 in order to use day_abbr method
df_train['DayOfWeek'] = df_train['DayOfWeek'].apply(lambda x: calendar.day_abbr[x])

#loading the datafields
url3 = 'https://drive.google.com/file/d/12zSselmNJapyFlv6anUpB7OYoonlOe7i/view?usp=sharing'
path3 = 'https://drive.google.com/uc?export=download&id='+url3.split('/')[-2]
df_datafields=pd.read_csv(path3)

#writing on the sidebar
url = 'https://drive.google.com/file/d/1qcwhWj62I_t7qApTgIzsbZ_ftLsOHq05/view?usp=sharing'
msba_logo = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#msba_logo=Image.open("C:\\Users\\Abou Assaly\\OneDrive - American University of Beirut\\AUB\\Spring\\MSBA 370 - Data Driven Digital Marketing\\Streamlit Final\\MSBA.png")
st.sidebar.image(msba_logo, use_column_width='auto')
html = '''
<p style="text-align: center; font-size: 15px">My name is Jad Abou Assaly, a Computer Engineer with a Masters in Business Analytics.
Being a Tech Savvy and a Data Science enthusiast, I strive to deliver <b>Data driven</b> solutions for end users.</p>
'''
st.sidebar.markdown(html, unsafe_allow_html=True)

st.sidebar.title("Explore")

option = st.sidebar.radio( '',
        ('Home','Data Exploration and visualization', 'Data Prediction'))

if option == 'Home':
    st.markdown("<h1 style='text-align: center'>Rossmann Store Analysis</h1>", unsafe_allow_html=True)
    url = 'https://drive.google.com/file/d/1bHG9kXMAoN20wSvg7xA2hszsBXsY4_ug/view?usp=sharing'
    analytics = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    #analytics=Image.open("C:\\Users\\Abou Assaly\\OneDrive - American University of Beirut\\AUB\\Spring\\MSBA 370 - Data Driven Digital Marketing\\Streamlit Final\\analytics.jpg")
    st.image(analytics, use_column_width='always')
    st.subheader("Overview")
    st.write("""This webapp aims to analyse and visualize the store sales of Rossmann drug stores which are influenced
    by multiple factors. It also has the capability of predicting the sales based on predefined parameters.
    The data is extracted from a Kaggle competition. """)

    st.subheader("Description")
    st.write("The table below shows the sales data of 1155 Rossman Drug Stores during 2013 and 2014")
    col1,col2=st.beta_columns(2)
    button1 = col1.button("View Table Header")
    if(button1):
        st.table(df_train.head())
    button2 = col2.button("View Fields Description")
    if(button2):
        st.table(df_datafields)



if option == 'Data Exploration and visualization':
    st.markdown("<h2 style='text-align: center'>Sales Exploration & visualization</h2>", unsafe_allow_html=True)
    expander = st.beta_expander("Overview")
    expander.write("This Section provides a variety of reports on a yearly, quarterly, monthly and daily basis. Please refer to the manual for extra information.")
    year = st.selectbox("Select a Year",(2013,2014))
    df_year=df_train[df_train['Year']==year]
    report=st.selectbox("Select a Report",("Top Sales Performance","Sales Performance per Store","Customer Visits per Store",
    "Customers & Sales","Store Type & Sales"))
    if report == "Top Sales Performance" :
        n=st.slider(f'Check the top performing stores in {year}', min_value=0, max_value=25,value=5)
        col1, col2, col3 = st.beta_columns(3)
        b1 = col1.button('Yearly')
        b2 = col2.button('Quarterly')
        b3 = col3.button('Monthly')

        if b1:
            #top n stores in yearly sales this year
            #n=st.slider(f'Check the top performing stores in yearly sales in {year}', min_value=0, max_value=25,value=5)
            df_perf=df_year.groupby(['Store'],as_index=False)['Sales'].sum()
            fig1 = px.bar(df_perf.nlargest(n,"Sales"), x="Sales", y="Store",orientation="h" ,title=f'Top {n} Performing Stores in Yearly Sales in {year}')
            st.plotly_chart(fig1)

        if b3:
            #top n stores in monthly sales this year
            #n1=st.slider(f'Check the top performing stores in monthly sales in {year}', min_value=0, max_value=25,value=5)
            df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Sales'].sum()
            df_highestn=df_monhtly.nlargest(n,'Sales')
            df_highestn['Store/Month']="Store:" + df_highestn['Store']+ " , Month:" +df_highestn['Month']
            fig2=px.bar(df_highestn,x='Sales',y='Store/Month',orientation='h',title=f'Top {n} Performing Stores in Monthly Sales in {year}')
            st.plotly_chart(fig2)

        if b2:
            #top n stores in quarterly sales this year
            #n2=st.slider(f'Check the top performing stores in quarterly sales in {year}', min_value=0, max_value=25,value=5)
            df_quarterly=df_year.groupby(['Store','Quarter'],as_index=False)['Sales'].sum()
            df_highestn2=df_quarterly.nlargest(n,'Sales')
            df_highestn2['Store/Quarter']="Store:" + df_highestn2['Store']+ " , Quarter:" +df_highestn2['Quarter']
            fig3=px.bar(df_highestn2,x='Sales',y='Store/Quarter',orientation='h',title=f'Top {n} Performing Stores in Quarterly Sales in {year}')
            st.plotly_chart(fig3)




    if report == "Sales Performance per Store":
        #Close look at each store performance
        s=st.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:
            time = st.selectbox("Select a Period",["Quarterly","Monthly","Daily"])
            if time == "Monthly":
                #monthly
                df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Sales'].sum()
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x))
                df_monhtly = df_monhtly.sort_values('to_sort')
                df_s=df_monhtly[df_monhtly['Store']==s]
                fig4 = px.line(df_s, x="Month", y="Sales", title=f'Sum of Monthly Sales of Store {s} in {year}')
                st.plotly_chart(fig4)

            if time == "Quarterly":
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter'],as_index=False)['Sales'].sum()
                df_s1=df_quarter[df_quarter['Store']==s]
                fig5=px.line(df_s1,x='Quarter',y='Sales',title=f'Sum of Quarterly Sales of Store {s} in {year}')
                st.plotly_chart(fig5)

            if time == "Daily":
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek'],as_index=False)['Sales'].mean()
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x))
                df_daily = df_daily.sort_values('to_sort')
                df_s2=df_daily[df_daily['Store']==s]
                fig6=px.line(df_s2,x='DayOfWeek',y='Sales',title=f'Average of Daily Sales of Store {s} in {year}')
                st.plotly_chart(fig6)

    if report == "Customer Visits per Store":
        #Close look at each store performance
        s=st.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:
            time = st.selectbox("Select a Period",["Quarterly","Monthly","Daily"])
            if time == "Monthly":
                #monthly
                df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Customers'].sum()
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x))
                df_monhtly = df_monhtly.sort_values('to_sort')
                df_s=df_monhtly[df_monhtly['Store']==s]
                fig7 = px.line(df_s, x="Month", y="Customers", title=f'Sum of Monthly Customers of Store {s} in {year}')
                st.plotly_chart(fig7)

            if time == "Quarterly":
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter'],as_index=False)['Customers'].sum()
                df_s1=df_quarter[df_quarter['Store']==s]
                fig8=px.line(df_s1,x='Quarter',y='Customers',title=f'Sum of Quarterly Customers of Store {s} in {year}')
                st.plotly_chart(fig8)

            if time == "Daily":
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek'],as_index=False)['Customers'].mean()
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x))
                df_daily = df_daily.sort_values('to_sort')
                df_s2=df_daily[df_daily['Store']==s]
                fig9=px.line(df_s2,x='DayOfWeek',y='Customers',title=f'Average of Daily Customers of Store {s} in {year}')
                st.plotly_chart(fig9)

    if report == "Customers & Sales":
        #Close look at each store performance
        s=st.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:
            time = st.selectbox("Select a Period",["Quarterly","Monthly","Daily"])
            if time == "Monthly":
                #monthly
                df_monhtly=df_year.groupby(['Store','Month','Sales'],as_index=False)['Customers'].sum()
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x))
                df_monhtly = df_monhtly.sort_values('to_sort')
                df_s=df_monhtly[df_monhtly['Store']==s]
                fig10 = px.scatter(df_s, x='Sales', y="Customers", title=f'Customers vs Sum of Monthly Sales of Store {s} in {year}',trendline="ols",color='Month')
                st.plotly_chart(fig10)

            if time == "Quarterly":
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter','Sales'],as_index=False)['Customers'].sum()
                df_s1=df_quarter[df_quarter['Store']==s]
                fig11=px.scatter(df_s1,x='Sales',y='Customers',title=f'Customers vs Sum of Quarterly Sales of Store {s} in {year}',trendline="ols",color='Quarter')
                st.plotly_chart(fig11)

            if time == "Daily":
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek','Sales'],as_index=False)['Customers'].mean()
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x))
                df_daily = df_daily.sort_values('to_sort')
                df_s2=df_daily[df_daily['Store']==s]
                fig12=px.scatter(df_s2,x='Sales',y='Customers',title=f'Customers vs Average of Daily Sales of Store {s} in {year}',trendline="ols",color='DayOfWeek')
                st.plotly_chart(fig12)


    if report == "Store Type & Sales":
        term = st.selectbox("Select a Period",["Yearly","Quarterly","Monthly"])

        if term == "Yearly":
            ordered_stores = ["a", "b", "c", "d"] #writing month in order since plotly uses alphabetical order
            df_year['to_sort']=df_year['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_year = df_year.sort_values('to_sort')
            fig13 = px.box(df_year, x="StoreType", y="Sales",title=f"Store Type vs Sales in {year}")
            st.plotly_chart(fig13)

        if term == "Quarterly":
            quarter = st.selectbox("Select a Quarter",["1","2","3","4"])
            df_quarter=df_year[df_year['Quarter']==quarter]
            ordered_stores = ["a", "b", "c", "d"] #writing month in order since plotly uses alphabetical order
            df_quarter['to_sort']=df_quarter['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_quarter = df_quarter.sort_values('to_sort')
            fig14 = px.box(df_quarter, x="StoreType", y="Sales",title=f"Store Type vs Sales in Q{quarter} of {year}")
            st.plotly_chart(fig14)

        if term == "Monthly":
            month = st.selectbox("Select a Month",["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            df_month=df_year[df_year['Month']==month]
            ordered_stores = ["a", "b", "c", "d"] #writing month in order since plotly uses alphabetical order
            df_month['to_sort']=df_month['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_month = df_month.sort_values('to_sort')
            fig15 = px.box(df_month, x="StoreType", y="Sales",title=f"Store Type vs Sales in {month} {year}")
            st.plotly_chart(fig15)


if option == 'Data Prediction':
    st.markdown("<h2 style='text-align: center'>Sales Prediction</h2>", unsafe_allow_html=True)
    expander = st.beta_expander("Overview")
    expander.write("This Section provides sales predictions depending on the user input. Please refer to the manual for extra information.")
    url1 = 'https://drive.google.com/file/d/1XMaIVbV2l1WUKNQPvP7J1X1sbT1zRtY8/view?usp=sharing'
    path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
    df_train=pd.read_csv(path1)

    url2 = 'https://drive.google.com/file/d/1QcnGtZKHdhTjE-oBjvVeRh22aNdO35Nk/view?usp=sharing'
    path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
    df_store=pd.read_csv(path2)

    df_train['Year'] = pd.DatetimeIndex(df_train['Date']). year
    df_train['Month'] = pd.DatetimeIndex(df_train['Date']). month
    df_train['Quarter']=pd.DatetimeIndex(df_train['Date']).quarter
    df_train=df_train[df_train['Year']!=2015]
    df_train=df_train.join(df_store.set_index('Store'),on='Store')
    df_cleaned=df_train.drop(['Store','Date','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Year','Open','Quarter','StateHoliday','SchoolHoliday'],axis=1)
    df_cleaned_1=df_cleaned.dropna()
    X=pd.get_dummies(df_cleaned_1,drop_first=True)
    y=X['Sales']
    X=X.drop(['Sales'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model =DecisionTreeRegressor(max_depth= 8,random_state=42)
    model.fit(X_train, y_train)

    features=['DayOfWeek', 'Customers', 'Promo', 'Month', 'CompetitionDistance',
       'Promo2', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_b',
       'Assortment_c']
    DayOfWeek=st.number_input('Enter the Day of the Week from 1 to 7',value=1,min_value=1,max_value=7,step=1)
    Month=st.number_input('Enter the Month from 1 to 12',value=1,min_value=1,max_value=12,step=1)
    Customers=st.number_input('Enter the number of customers',value=100,step=1)
    Promo=st.number_input('Is the store running a promo on this day? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    Promo2=st.number_input('Is there a continuing promotion? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    CompetitionDistance=st.number_input('Enter a distance in meters to the nearest competitor store',value=1000,step=1)
    StoreType_b=st.number_input('Is it a type b store? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    StoreType_c=st.number_input('Is it a type c store? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    StoreType_d=st.number_input('Is it a type d store? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    Assortment_b=st.number_input('Is it Assortment Level b? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)
    Assortment_c=st.number_input('Is it Assortment Level c? 0=no, 1=yes',value=0,min_value=0,max_value=1,step=1)

    predictions = model.predict([[DayOfWeek, Customers, Promo, Month, CompetitionDistance,
       Promo2, StoreType_b, StoreType_c, StoreType_d, Assortment_b,
       Assortment_c]])
    prediction = round(predictions[0],2)
    st.write(f"The predicted turnover of this store given your input is {prediction} USD")
