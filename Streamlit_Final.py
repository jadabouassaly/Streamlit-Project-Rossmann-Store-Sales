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

st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

show_streamlit_style = """
            <style>
            footer:after {
            	content:'Developed By J.A.A.';
            	visibility: visible;}
            </style>
            """
st.markdown(show_streamlit_style, unsafe_allow_html=True)


#function to read the data with caching
@st.cache(allow_output_mutation=True)
def load_data(path):
    df=pd.read_csv(path)
    return df

#read the 1st dataframe
url1 = 'https://drive.google.com/file/d/1XMaIVbV2l1WUKNQPvP7J1X1sbT1zRtY8/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
df_train=load_data(path1)

#read the 2nd dataframe
url2 = 'https://drive.google.com/file/d/1QcnGtZKHdhTjE-oBjvVeRh22aNdO35Nk/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
df_store=load_data(path2)


@st.cache
def transform(df,df1):
    df['Year'] = pd.DatetimeIndex(df['Date']).year #retreive the year from the date
    df['Month'] = pd.DatetimeIndex(df['Date']).month #retreive the month from the date
    df['Quarter']=pd.DatetimeIndex(df['Date']).quarter #retreive the quarter from the date
    df=df[df['Year']!=2015] #drop year 2015 since it is incomplete
    df=df.join(df1.set_index('Store'),on='Store') #join both datasets on "store"
    df['Store']=df['Store'].astype(str) #convert store to string
    df['Quarter']=df['Quarter'].astype(str) #convert quarter to string
    df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x]) #convert the month number to its abbreviation

    #converting DayOfWeek from number to day name
    df['DayOfWeek'] = df['DayOfWeek'] -1 #days should be from 0 to 6 in order to use day_abbr method
    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: calendar.day_abbr[x])
    return df

df_train=transform(df_train,df_store)

url3 = 'https://drive.google.com/file/d/12zSselmNJapyFlv6anUpB7OYoonlOe7i/view?usp=sharing'
path3 = 'https://drive.google.com/uc?export=download&id='+url3.split('/')[-2]
df_datafields=load_data(path3)

#writing on the sidebar
url = 'https://drive.google.com/file/d/1qcwhWj62I_t7qApTgIzsbZ_ftLsOHq05/view?usp=sharing'
msba_logo = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
st.sidebar.image(msba_logo, use_column_width='auto')


html = '''
<p style="text-align: center; font-size: 15px">My name is Jad Abou Assaly, a Computer Engineer with a Masters in Business Analytics.
Being a Tech Savvy and a Data Science enthusiast, I strive to deliver <b>Data driven</b> solutions for end users.</p>
<hr class="rounded">
'''
st.sidebar.markdown(html, unsafe_allow_html=True)

st.sidebar.title("Explore Preloaded Data")

option = st.sidebar.radio( '',
        ('Home','Data Exploration and Visualization', 'Sales Prediction'))

#adding a line in html
html = '''
<hr class="rounded">
'''
st.sidebar.markdown(html, unsafe_allow_html=True)

st.sidebar.title("Upload Your Data")
data = st.sidebar.file_uploader('Upload your Data in csv format and make sure to fill all the Described Fields.',type=['csv'])


st.sidebar.markdown(html, unsafe_allow_html=True)


html = '''
<p style="text-align: center; font-size: 15px">For additional info or enhancements, Kindly send your request to jsa48@mail.aub.edu</p>
'''
st.sidebar.markdown(html, unsafe_allow_html=True)


if option == 'Home':
    st.markdown("<h1 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Rossmann Store Analysis</h1>", unsafe_allow_html=True)
    url = 'https://drive.google.com/file/d/1bHG9kXMAoN20wSvg7xA2hszsBXsY4_ug/view?usp=sharing'
    analytics = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    st.image(analytics, use_column_width='always')



    st.subheader("Overview")
    st.write("""This webapp aims to analyse and visualize the store sales of Rossmann drug stores during 2013 and 2014.
    It also has the capability of predicting the sales based on predefined parameters.
    The preloaded data is extracted from a Kaggle competition. """)

    st.subheader("Description")
    st.write("The table below shows the sales data of 1115 Rossman Drug Stores during 2013 and 2014")
    col1,col2,col3=st.beta_columns([1,1,3])
    button1 = col1.button("View Fields Description")
    if(button1):
        st.table(df_datafields)
    button2 = col2.button("View Table Header")
    if(button2):
        st.table(df_train.head())

@st.cache
def transform1(df):
    df['Store']=df['Store'].astype(str)
    df['Quarter']=df['Quarter'].astype(str)
    df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
    #converting DayOfWeek from number to day name
    df['DayOfWeek'] = df['DayOfWeek'] -1 #days should be from 0 to 6 in order to use day_abbr method
    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: calendar.day_abbr[x])
    return df

if option == 'Data Exploration and Visualization':
    if data is not None: #if data is uploaded by the user
        df_train = pd.read_csv(data)
        df_train = transform1(df_train)



    st.markdown("<h1 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Sales Exploration & Visualization</h1>", unsafe_allow_html=True)
    expander = st.beta_expander("Overview")
    expander.write("This Section provides a variety of reports on a yearly, quarterly, monthly and daily basis. Please refer to the manual for extra information.")
    col1, col2, col3 = st.beta_columns([3,3,1])
    year = col3.selectbox("Select a Year",(df_train.Year.unique())) #the available values in the dropdown are the unique values in the "year" column
    df_year=df_train[df_train['Year']==year] #selecting the data corresponding to the selected year
    report=col1.selectbox("Select a Report",("Top Sales Performance","Sales Performance per Store","Customer Visits per Store",
    "Customers & Sales","Store Type & Sales"))


    if report == "Top Sales Performance" :
        n=col2.slider(f'Check the top performing stores in {year}', min_value=1, max_value=25,value=5)
        col1, col2, col3 = st.beta_columns([5,1,5])

        with col1:
            #top n stores in yearly sales this year
            df_perf=df_year.groupby(['Store'],as_index=False)['Sales'].sum() #group by store and retreiving the sum of sales of each
            fig1 = px.bar(df_perf.nlargest(n,"Sales"), x="Sales", y="Store",orientation="h" ,title=f'Top {n} Performing Store(s) in Yearly Sales in {year}')
            fig1.update_traces(marker_color='#DD0734')
            st.plotly_chart(fig1)


        with col3:
        #top n stores in quarterly sales this year
            df_quarterly=df_year.groupby(['Store','Quarter'],as_index=False)['Sales'].sum()#group by store and quarter and retreiving the sum of sales of each
            df_highestn2=df_quarterly.nlargest(n,'Sales') #retreiving the n largest
            df_highestn2['Store/Quarter']="Store:" + df_highestn2['Store']+ " , Quarter:" +df_highestn2['Quarter'] #creating a new column containing the corresponding store and quarter
            fig3=px.bar(df_highestn2,x='Sales',y='Store/Quarter',orientation='h',title=f'Top {n} Performing Store(s) in Quarterly Sales in {year}')
            st.plotly_chart(fig3)

        col1, col2, col3 = st.beta_columns([5,1,5])
        with col1:
            #top n stores in monthly sales this year
            df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Sales'].sum()#group by store and month and retreiving the sum of sales of each
            df_highestn=df_monhtly.nlargest(n,'Sales') #retreiving the n largest
            df_highestn['Store/Month']="Store:" + df_highestn['Store']+ " , Month:" +df_highestn['Month'] #creating a new column containing the corresponding store and month
            fig2=px.bar(df_highestn,x='Sales',y='Store/Month',orientation='h',title=f'Top {n} Performing Store(s) in Monthly Sales in {year}')
            fig2.update_traces(marker_color='#53565A')
            st.plotly_chart(fig2)

        with col3:
            st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Conclusion</h3>", unsafe_allow_html=True)
            x=df_perf.nlargest(1,"Sales").Store
            y=df_perf.nlargest(1,"Sales").Sales
            #st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>x</h3>", unsafe_allow_html=True)
            st.subheader(f'The best performing store in {year} is store number {x.iloc[0]} with sales equal to {y.iloc[0]} USD.')

    if report == "Sales Performance per Store":
        #Close look at each store performance
        s=col2.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:
            #time = st.selectbox("Select a Period",["Quarterly","Monthly","Daily"])
            col1, col2, col3 = st.beta_columns([4,1,5])
            #if time == "Monthly":
            with col1:
                #monthly
                df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Sales'].sum() #group by store and month and retreiving the sum of sales of each
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x)) #transforming month to index
                df_monhtly = df_monhtly.sort_values('to_sort') #sorting on month
                df_s=df_monhtly[df_monhtly['Store']==s] #filtering the requested store
                fig4 = px.line(df_s, x="Month", y="Sales", title=f'Sum of Monthly Sales of Store {s} in {year}')
                fig4.update_traces(line_color='#DD0734')
                st.plotly_chart(fig4)

            #if time == "Quarterly":
            with col3:
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter'],as_index=False)['Sales'].sum() #group by store and quarter and retreiving the sum of sales of each
                df_s1=df_quarter[df_quarter['Store']==s] #filtering the requested store
                fig5=px.line(df_s1,x='Quarter',y='Sales',title=f'Sum of Quarterly Sales of Store {s} in {year}')
                st.plotly_chart(fig5)

            col1, col2, col3 = st.beta_columns([5,1,5])
            #if time == "Daily":
            with col1:
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek'],as_index=False)['Sales'].mean() #group by store and day and retreiving the average sales of each
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x)) #transforming day to index
                df_daily = df_daily.sort_values('to_sort') #sorting on day
                df_s2=df_daily[df_daily['Store']==s] #filtering the requested store
                fig6=px.line(df_s2,x='DayOfWeek',y='Sales',title=f'Average of Daily Sales of Store {s} in {year}')
                fig6.update_traces(line_color='#53565A')
                st.plotly_chart(fig6)

            with col3:
                st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Conclusion</h3>", unsafe_allow_html=True)
                column=df_s["Sales"]
                max_sale = column.max()
                max_index = column.idxmax()
                max_month=df_s["Month"][max_index]
                column1=df_s2["Sales"]
                max_sale1 = round(column1.max(),2)
                max_index1 = column1.idxmax()
                max_day=df_s2["DayOfWeek"][max_index1]
                st.subheader(f'The best sales performance of store {s} in {year} is in {max_month} with sales equal to {max_sale} USD.')
                st.subheader(f'The best average daily sales falls on {max_day} with an amount of {max_sale1} USD.')






    if report == "Customer Visits per Store":
        #Close look at each store performance
        s=col2.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:
            col1, col2, col3 = st.beta_columns([4,1,5])
            with col1:
                #monthly
                df_monhtly=df_year.groupby(['Store','Month'],as_index=False)['Customers'].sum()
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x))
                df_monhtly = df_monhtly.sort_values('to_sort')
                df_s=df_monhtly[df_monhtly['Store']==s]
                fig7 = px.line(df_s, x="Month", y="Customers", title=f'Sum of Monthly Customers of Store {s} in {year}')
                fig7.update_traces(line_color='#DD0734')
                st.plotly_chart(fig7)

            with col3:
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter'],as_index=False)['Customers'].sum()
                df_s1=df_quarter[df_quarter['Store']==s]
                fig8=px.line(df_s1,x='Quarter',y='Customers',title=f'Sum of Quarterly Customers of Store {s} in {year}')
                st.plotly_chart(fig8)


            col1, col2, col3 = st.beta_columns([5,1,5])
            with col1:
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek'],as_index=False)['Customers'].mean()
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x))
                df_daily = df_daily.sort_values('to_sort')
                df_s2=df_daily[df_daily['Store']==s]
                fig9=px.line(df_s2,x='DayOfWeek',y='Customers',title=f'Average of Daily Customers of Store {s} in {year}')
                fig9.update_traces(line_color='#53565A')
                st.plotly_chart(fig9)

            with col3:
                st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Conclusion</h3>", unsafe_allow_html=True)
                column=df_s["Customers"]
                max_cust = column.max()
                max_index = column.idxmax()
                max_month=df_s["Month"][max_index]
                column1=df_s2["Customers"]
                max_cust1 = round(column1.max(),0)
                max_index1 = column1.idxmax()
                max_day=df_s2["DayOfWeek"][max_index1]
                st.subheader(f'The highest number of store {s} customers in {year} is in {max_month} with a total equal to {max_cust} customers.')
                st.subheader(f'On a daily basis, the highest average customer visits falls on {max_day} with an average of {max_cust1} customers.')



    if report == "Customers & Sales":
        #Close look at each store performance
        s=col2.text_input('Enter store number between 1 and 1115',value="1") #default value to 1
        s1=int(s) #coverting to int to apply the condition
        if s1>1115 or s1<1:
            st.write("Enter store number between 1 and 1115")
        else:

            col1, col2 = st.beta_columns([1,1])
            with col1:
                #monthly
                df_monhtly=df_year.groupby(['Store','Month','Sales'],as_index=False)['Customers'].sum()
                ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #writing month in order since plotly uses alphabetical order
                df_monhtly['to_sort']=df_monhtly['Month'].apply(lambda x:ordered_months.index(x))
                df_monhtly = df_monhtly.sort_values('to_sort')
                df_s=df_monhtly[df_monhtly['Store']==s]
                fig10 = px.scatter(df_s, x='Customers', y="Sales", title=f'Customers v/s Sum of Monthly Sales of Store {s} in {year}',trendline="ols",color='Month')
                st.plotly_chart(fig10)

            with col2:
                #quaterly
                df_quarter=df_year.groupby(['Store','Quarter','Sales'],as_index=False)['Customers'].sum()
                df_s1=df_quarter[df_quarter['Store']==s]
                fig11=px.scatter(df_s1,x='Customers', y="Sales",title=f'Customers v/s Sum of Quarterly Sales of Store {s} in {year}',trendline="ols",color='Quarter')
                st.plotly_chart(fig11)

            col1, col2, col3 = st.beta_columns([5,1,5])
            with col1:
                #daily
                df_daily=df_year.groupby(['Store','DayOfWeek','Sales'],as_index=False)['Customers'].mean()
                ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] #writing month in order since plotly uses alphabetical order
                df_daily['to_sort']=df_daily['DayOfWeek'].apply(lambda x:ordered_days.index(x))
                df_daily = df_daily.sort_values('to_sort')
                df_s2=df_daily[df_daily['Store']==s]
                fig12=px.scatter(df_s2,x='Customers', y="Sales",title=f'Customers v/s Average of Daily Sales of Store {s} in {year}',trendline="ols",color='DayOfWeek')
                st.plotly_chart(fig12)

            with col3:
                st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Hint</h3>", unsafe_allow_html=True)
                #st.markdown("<h4 style='color:#53565A'>Double click on the graph legend to isolate the desired period. You can also hover over the graphs to check the trendline showing the relationship between the Sales amounts and Customers volume.</h4>", unsafe_allow_html=True)
                st.subheader('Double click on the graph legend to isolate the desired period. You can also hover over the graphs to check the trendline showing the relationship between the Sales amounts and Customers volume.')

    if report == "Store Type & Sales":
        #term = col2.selectbox("Select a Period",["Yearly","Quarterly","Monthly"])
        col1, col2, col3 = st.beta_columns([3,3,1])
        quarter = col3.selectbox("Select a Quarter",["1","2","3","4"]) #select the quarter
        col1, col2, col3 = st.beta_columns([3,3,1])
        month = col3.selectbox("Select a Month",["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]) #select the month

        col1, col2 = st.beta_columns([3,3])
        with col1:
            ordered_stores = ["a", "b", "c", "d"] #writing store types in order since plotly uses alphabetical order
            df_year['to_sort']=df_year['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_year = df_year.sort_values('to_sort')
            x=df_year.groupby(['StoreType'],as_index=False)['Sales'].sum()
            fig13 = px.bar(x, x="StoreType", y="Sales",title=f"Store Type v/s Sales in {year}")
            fig13.update_traces(marker_color='#DD0734')
            st.plotly_chart(fig13)

        with col2:

            df_quarter=df_year[df_year['Quarter']==quarter] #filter based on the chosen quarter
            ordered_stores = ["a", "b", "c", "d"] #writing store types in order since plotly uses alphabetical order
            df_quarter['to_sort']=df_quarter['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_quarter = df_quarter.sort_values('to_sort')
            y=df_quarter.groupby(['StoreType'],as_index=False)['Sales'].sum()
            fig14 = px.bar(y, x="StoreType", y="Sales",title=f"Store Type v/s Sales in Q{quarter} of {year}")
            st.plotly_chart(fig14)

        col1, col2,col3 = st.beta_columns([3,1,3])
        with col1:

            df_month=df_year[df_year['Month']==month] #filter based on the chosen month
            ordered_stores = ["a", "b", "c", "d"] #writing store types in order since plotly uses alphabetical order
            df_month['to_sort']=df_month['StoreType'].apply(lambda x:ordered_stores.index(x))
            df_month = df_month.sort_values('to_sort')
            z=df_month.groupby(['StoreType'],as_index=False)['Sales'].sum()
            fig15 = px.bar(z, x="StoreType", y="Sales",title=f"Store Type v/s Sales in {month} {year}")
            fig15.update_traces(marker_color='#53565A')
            st.plotly_chart(fig15)

        with col3:
            st.markdown("<h3 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Conclusion</h3>", unsafe_allow_html=True)
            column=x["Sales"]
            max_sale = column.max()
            max_index = column.idxmax()
            max_store=x["StoreType"][max_index]
            column1=y["Sales"]
            max_sale1 = column1.max()
            max_index1 = column1.idxmax()
            max_store1=y["StoreType"][max_index1]
            column2=z["Sales"]
            max_sale2 = column2.max()
            max_index2 = column2.idxmax()
            max_store2=z["StoreType"][max_index2]
            st.subheader(f'Store type {max_store} has the highest yearly sales figures in {year} with sales reaching {max_sale} USD.')
            st.subheader(f'Store type {max_store1} has the highest quarterly sales figures in quarter Q{quarter} in {year} with sales reaching {max_sale1} USD.')
            st.subheader(f'Store type {max_store2} has the highest monthly sales figures in {month} {year} with sales reaching {max_sale2} USD.')

@st.cache
def clean_fit(df,df1):
    df['Year'] = pd.DatetimeIndex(df['Date']). year
    df['Month'] = pd.DatetimeIndex(df['Date']). month
    df['Quarter']=pd.DatetimeIndex(df['Date']).quarter
    df=df[df['Year']!=2015]
    df=df.join(df1.set_index('Store'),on='Store')

    #dropping the unwanted columns
    df_cleaned=df.drop(['Store','Date','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Year','Open','Quarter','StateHoliday','SchoolHoliday'],axis=1)
    df_cleaned_1=df_cleaned.dropna()

    #getting the features and the target
    X=pd.get_dummies(df_cleaned_1,drop_first=True)
    y=X['Sales']
    X=X.drop(['Sales'],axis=1)

    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #choosing the model and fitting it on the train data
    model =DecisionTreeRegressor(max_depth= 8,random_state=42)
    fitted_model=model.fit(X_train, y_train)
    return fitted_model

if option == 'Sales Prediction':
    st.markdown("<h1 style='text-align: center;background-color:#F0F2F6;color:#53565A'>Sales Prediction</h1>", unsafe_allow_html=True)
    expander = st.beta_expander("Overview")
    expander.write("This Section provides sales predictions depending on the user input. Please refer to the manual for extra information.")

    #loading the data again and applying the transformations, however this time i dropped some unwanted columns based on feature
    #importance study performed seperately.
    url1 = 'https://drive.google.com/file/d/1XMaIVbV2l1WUKNQPvP7J1X1sbT1zRtY8/view?usp=sharing'
    path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
    df_train=load_data(path1)

    url2 = 'https://drive.google.com/file/d/1QcnGtZKHdhTjE-oBjvVeRh22aNdO35Nk/view?usp=sharing'
    path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
    df_store=load_data(path2)

    model=clean_fit(df_train,df_store)

    StoreType_b=StoreType_c=StoreType_d=Assortment_b=Assortment_c=0

    #asking the user to input the needed fields
    col1,col2,col3=st.beta_columns(3)
    DayOfWeek=col1.number_input('Enter the Day of the Week from 1 to 7',value=1,min_value=1,max_value=7,step=1)
    Month=col2.number_input('Enter the Month from 1 to 12',value=1,min_value=1,max_value=12,step=1)
    Customers=col3.number_input('Enter the number of customers',value=100,step=1)

    col1,col2,col3=st.beta_columns(3)
    CompetitionDistance=col1.number_input('Enter a distance in meters to the nearest competitor store',value=1000,step=1)
    StoreType=col2.selectbox('Select the store type',['a','b','c','d'])
    if StoreType == 'b':
        StoreType_b=1

    elif StoreType =='c':
        StoreType_c=1

    elif StoreType =='d':
        StoreType_d=1

    AssortmentType=col3.selectbox('Select the assortment type',['a','b','c'])
    if AssortmentType == 'b':
        Assortment_b=1

    elif AssortmentType =='c':
        Assortment_c=1

    col1,col2,col3=st.beta_columns(3)
    Promo=col1.checkbox('The store is running a promo')
    Promo2=col2.checkbox('The store is running a continuing promo')


    st.write("#")
    st.write("#")


    col1,col2,col3=st.beta_columns(3)
    #predict based on the fitted model
    b=col1.button('Predict Store Sales')
    if (b):
        predictions = model.predict([[DayOfWeek, Customers, Promo, Month, CompetitionDistance,Promo2, StoreType_b, StoreType_c, StoreType_d, Assortment_b,Assortment_c]])
        prediction = round(predictions[0],2)
        st.subheader(f"The predicted turnover of this store given your input is **{prediction}** USD")
