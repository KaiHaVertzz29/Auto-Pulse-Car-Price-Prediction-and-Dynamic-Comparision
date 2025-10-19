from calendar import c
import numpy as np
from email.policy import default
import glob
from logging import PlaceHolder
from re import S
import streamlit as st
import pandas as pd
import numpy as npf
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from datetime import datetime
import pickle
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plost
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.set_page_config(layout="wide")
    menu = ["Compare", "Predict","Download"]
    col1,col2=st.columns((3,7))
    choice = col1.selectbox("Navigate To Feature",menu,index=None)
    
    if choice=='Compare':
        with col2:
            col11,col22=st.columns(2)
            option=col22.selectbox('Select Comparision Type : ',['Comparision Tab','Insights Tab'],index=None)

    dynamic_content = st.empty()
    dynamic_content.markdown("""
            # :orange[Auto Pulse]
            ###### Precision Car Pricing and Dynamic Comparison            
            -------------------------------------------------
            * :blue[**Navigate to Compare page for :** ]
                * exploring database in csv format
                * inter/intra brand comparisions
                * comparision through visualization.
            
            --------------------------------------------------

            * :red[**Navigate to Predict Page for :**]
                * predicting price of your car 
                * Some primary details of the car such as year of manufacture, car brand , car type needs to selected before predicting 
                
                """)
    
    if choice == "Predict":
        dynamic_content.text("")
        home_page()
    
    elif choice == "Compare":
        dynamic_content.text("")
        compare_page(option)

    elif choice == 'Download':
        dynamic_content.text("")
        download_page()
    
def compare_page(option):

    dynamic_content_2=st.empty()
    dynamic_content_2.markdown("""
                               # :orange[Visualization Spot]
                             ##### **Select pages from sidebar to go to different visualization options**
                               * select :blue[comaprision tab] for comparing data variables and get insights
                               * select :green[data visualized] tab for display datasets in visualizations
                             """)
    

    if option=='Insights Tab':
        dynamic_content_2.text("")
        data_visualized()
    elif option=='Comparision Tab':
        dynamic_content_2.text("")
        comparision_tab()
                
def home_page():

    st.markdown("""
                # :blue[Auto Pulse]
            Precision Car Pricing and Dynamic Comparison
            * **Primary selections required :** condition , year of manufacture , car type, car brand and car model
            * **Optional selections:** mileage, drive type, transmission, interior and exterior color
            * The filters are filled with default values change and click on predict button below to get your price 
            """)
    def expensive_computation(input_data):
    # Perform some expensive computation here
        result = input_data + 1
        return result

    st.sidebar.markdown("""## :orange[Select Specifics Here]""")

    with open('/Users/dhruvparikh/Downloads/Untitled Folder/forest_model.pkl', 'rb') as file:
        price_predictor=pickle.load(file)

    with open('/Users/dhruvparikh/Downloads/Untitled Folder/dtree.pkl', 'rb') as file:
        dtree_price_predictor=pickle.load(file)


    car_data=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/car_data.csv')

    by_brand=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/by_brand.csv')

    logo_links=pd.read_csv('/Users/dhruvparikh/Downloads/car_logos.csv')

    model_data=car_data[['origin','condition','car_model','mileage','exterior_color',
                'interior_color','seating_capacity','transmission',
                'drive_type','price_dollars','engine_type','engine_capacity','age',
                'brand_code','brand_grade']]

    other_grades=car_data[~(car_data.grade=='other')].brand.unique()+'_'+'other'
    brand_grade_other=pd.DataFrame(np.concatenate((car_data.brand_grade.unique(),other_grades)))
    brand_grade_other=brand_grade_other.rename(columns={0:'brand_grade'})
    brand_grade_other.loc[brand_grade_other.shape[0],'brand_grade']='other_other' 

    brand_grade_encoder=OneHotEncoder(drop='first')
    brand_grade_encoder.fit(brand_grade_other[['brand_grade']]) 

    encoder_except_brand_grade=OneHotEncoder(drop='first')
    encoder_except_brand_grade.fit(model_data.drop(columns='brand_grade').select_dtypes(include='object'))

    scaler=MinMaxScaler()
    model_data['mileage']=scaler.fit_transform(model_data[['mileage']])

   

    def encode_model_data(df):
        
        df['mileage']=scaler.transform(df[['mileage']])

        df['brand_code']=df.brand_code.astype('int64')

        df.brand_code.fillna(3)

        brand_grade_encoded=brand_grade_encoder.transform(df[['brand_grade']]).toarray()
        
        others_encoded=encoder_except_brand_grade.transform(df.drop(columns='brand_grade').select_dtypes(include='object')).toarray()
        
        data=pd.concat(
        [df.select_dtypes(exclude='object').reset_index(drop=True),
        pd.DataFrame(others_encoded,columns=encoder_except_brand_grade.get_feature_names_out()),
        pd.DataFrame(brand_grade_encoded,columns=brand_grade_encoder.get_feature_names_out())],axis=1)
        
        
        return(data)




    def get_default_dataframe(df,selected_brand,selected_model,selected_type,transmission,interior_color,exterior_color,mileage,drive_type,year,condition):
        columns=[['origin','condition','car_model','mileage','exterior_color',
                    'interior_color','seating_capacity','transmission',
                    'drive_type','engine_type','engine_capacity','age',
                    'brand_code','brand_grade']]
        current_year=datetime.now().year
        brand_grade=selected_brand+'_'+selected_model
        seating_capacity=df[df.brand_grade==brand_grade].seating_capacity.mean()
        engine_capcaity=df[df.brand_grade==brand_grade].engine_capacity.mean()-0.5
        engine_type=df[df.brand_grade==brand_grade].engine_type.agg(pd.Series.mode)[0]
        brand_code=by_brand[by_brand.brand==selected_brand].encoded
        origin=df[df.brand_grade==brand_grade].origin.agg(pd.Series.mode)[0]
        age=current_year-int(year)

        prediction_series={'origin':origin,'condition':condition,
                        'car_model':selected_type,'mileage':mileage,'exterior_color':exterior_color,
                        'interior_color':interior_color,'seating_capacity':seating_capacity,
                        'transmission':transmission,'drive_type':drive_type,
                        'engine_type':engine_type,'engine_capacity':engine_capcaity,
                        'age':age,
                        'brand_code':brand_code,'brand_grade':brand_grade
                        }
        
        return(encode_model_data(pd.DataFrame([prediction_series])))
        



    with st.form('my_form'):
    
        condition=st.sidebar.radio('Condition',['New car','Used car'])
        year=st.sidebar.slider('Year Of Manufacture',car_data[car_data.condition==condition].year_of_manufacture.min(),datetime.now().year,key='condition_selector')
        
        type=tuple(car_data.car_model.unique())
        car_type=st.sidebar.selectbox('Car_Type',type,key='type')
        
        brands=tuple(car_data[car_data.car_model==car_type].brand.unique())
        car_brand=st.sidebar.selectbox('Brand',brands,key='brands')
        
        model=tuple(car_data[(car_data.brand==car_brand)&(car_data.car_model==car_type)].grade.unique())
        car_model=st.sidebar.selectbox('Model',model,key='models')
        
        st.sidebar.write('\n :bold :red[Additional Filters (can be skipped)]\t :point_down:')
        mileage=st.sidebar.slider('Mileage',0,1000000,step=100,key='mileage')
        
        color=list(car_data[car_data.brand_grade==car_brand+'_'+car_model].exterior_color.unique())
        if 'Take note' in color:
            color.remove('Take note')
            color.append('Other')
        
        else:
            color.append('Other')
            
        exterior_color=st.sidebar.selectbox('Exterior Color',color,key='color')
        
        interior=list(car_data[car_data.brand_grade==car_brand+'_'+car_model].interior_color.unique())
        interior_color=st.sidebar.selectbox('interior Color',interior,key='interior')
        
        transmission=tuple(car_data[car_data.brand_grade==car_brand+'_'+car_model].transmission.unique())
        transmission=st.sidebar.selectbox('Transmission',transmission,key='trans')
        
        drive_type=tuple(car_data[car_data.brand_grade==car_brand+'_'+car_model].drive_type.unique())
        type_drive=st.sidebar.selectbox('Drive Type',drive_type,key='drive_type')

        submitted=st.form_submit_button(":red[Predict \t\t Price \t\t ] :oncoming_automobile:")


    if submitted:
        link_name=list(logo_links[logo_links.Brand.str.lower()==str.lower(car_brand)].Link)
        st.image(link_name[0], caption=car_brand,width=100)
        predict=get_default_dataframe(car_data,car_brand,car_model,car_type,transmission,interior_color,exterior_color,mileage,type_drive,year,condition)
        st.markdown(f'''
                    * Predicted Value for **{car_brand+' - '+car_model}** is &nbsp; :red[**$ {str(np.round(price_predictor.predict(predict),2)).strip('[]')}**]''')

def data_visualized():
    st.markdown('''
            # :blue[Insights Tab]
             Visual Insigts Of Our Datset
                ''')
    
    car_data=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/car_data.csv')
    
    req_data=car_data[['origin','condition','car_model','mileage','exterior_color',
                'interior_color','seating_capacity','transmission',
                'drive_type','price_dollars','engine_type','engine_capacity','year_of_manufacture',
                'brand_code','brand_grade','brand','grade']]
    
    by_brand=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/by_brand.csv')
    
    logo_links=pd.read_csv('/Users/dhruvparikh/Downloads/car_logos.csv')


    st.markdown('* :red[Select variables from below bar to get visualized insights]')
    selection=st.selectbox('Select',['car_model','exterior_color','brand','Numerical Data'],index=None)
    col1,col2=st.columns((5,5))
    
    if selection=='car_model':
        
        with col1:
            plost.bar_chart(
                req_data.groupby('car_model').price_dollars.mean().reset_index(),
                bar='car_model',
                value='price_dollars',
                direction='horizontal',
                use_container_width=True,
            )

            st.markdown('''  
                         ###### :blue[Knowing which type of car is more in the market : ] 
                        \n
                         * SUV and Sedan cover one third of the market each making it the most demanded car type
                        \n
                         * pickup , truck and van take up less than 10% market space making them the least preferred type
                        \n
                        * others share equal amount of market space  
                          
                        --------------------------------
                    ''')
            
            st.markdown('###### :orange[Select type to get brands share :]')
            selected_model=st.selectbox('Select :' ,tuple(req_data.car_model.unique()),index=0)

            st.markdown("""  
                    * select car type from above to get the share that the brands share of this car type
                        \n
                    * This is only subjective to my dataset . Global figures may vary
                        
                        """)


        with col2:
            
            st.markdown(''' 
                            ###### :red[**Some points that can be extracted are :**]
                        \n
                        * convertible and coupe cars tends to have higher price than other car models/types
                        \n
                        * pickup trucks , hatchback and wagon has lower prices mainly because these donot contain any luxury brands and tend to have similar average prices
                        \n
                        ----------------------------
                        --------------------------------------
                        ''')
            
            plost.donut_chart(
                (np.round(req_data.car_model.value_counts()*100/req_data.shape[0],2)).reset_index(),
                theta='count',
                color='car_model',
                height=230,
                title='.',
                use_container_width=True

            )

            if selected_model:
                plost.pie_chart(
                    (np.round(req_data[req_data.car_model==selected_model]\
                             .brand.value_counts()*100/req_data[req_data.car_model==selected_model].shape[0],2)).reset_index(),
                    theta='count',
                    color='brand',
                    height=300,
                    title='.',
                    use_container_width=True)


    
            
    elif selection=='exterior_color':

        st.markdown("""
                    * :red[Average price for each color]
                    * Doesnot take universal trends just shows the dataset values """)

        plost.bar_chart(
            req_data.groupby('exterior_color').price_dollars.mean().reset_index(),
            bar='exterior_color',
            value='price_dollars',
            use_container_width=True,
            color='exterior_color'
        )

        

        st.markdown(""" ##### :orange[Market share of colors :]
                    \n""")

        plost.pie_chart(
            (np.round(req_data.exterior_color.value_counts()*100/req_data.shape[0],2)).reset_index(),
            theta='count',
            color='exterior_color',
            use_container_width=True
            
        )

        st.markdown('* hover values are in percentage')

    elif selection=='brand':

        st.markdown('* Average brand price (according to our dataset not universal stats)')

        plost.bar_chart(
            req_data.groupby('brand').price_dollars.mean().reset_index(),
            bar='brand',
            value='price_dollars',
            use_container_width=True,
            color='price_dollars'
        )
        
        

        price_ranges = [0, 30000, 60000, 100000, 200000, float('inf')]
        labels = ['Entry-Level', 'Mid-Range', 'Premium/Executive', 'Luxury/High-End', 'Ultra-Luxury/Exotic']

        req_data['coded_brand']=pd\
            .cut(car_data.price_dollars,bins=price_ranges,labels=labels)
        
        labels_encoded={'Entry-Level':1, 'Mid-Range':2, 'Premium/Executive':3, 'Luxury/High-End':4, 'Ultra-Luxury/Exotic':5}
        req_data['labels_encoded']=req_data.coded_brand.map(labels_encoded)
        req_data['labels_encoded']=req_data.labels_encoded.astype('int64')

        st.markdown(" ##### :orange[Below is a chart showing brands with their price score] ")

        st.markdown(''' 
                    ###### Brand score is the metric used for below chart where :
                    * :red[Entry-Level] : 1
                    * :orange[Mid-Range] : 2
                    * :grey[Premium/Executive] : 3
                    * :green[Luxury/High-End] : 4
                    * :green[Ultra Luxury] : 5
                    
                        ''')

        plost.bar_chart(
            np.round(req_data.groupby('brand').labels_encoded.mean(),2).reset_index(),
            bar='brand',
            value='labels_encoded',
            use_container_width=True,
            color='labels_encoded'
        )
        
        st.markdown('##### :red[Get Brands Categorization per price Score]')

        brand_encoded=np.round(req_data.groupby('brand').labels_encoded.mean(),0).reset_index()
        df = pd.DataFrame(list(labels_encoded.items()), columns=['category', 'value'])
        df = df.reset_index(drop=True)
        plot_data=brand_encoded.merge(df,left_on='labels_encoded',right_on='value',how='left')

        type_select=st.selectbox('Select Type :',labels)
        col1,col2=st.columns((3,7))
        
        data_req=req_data[req_data.brand.isin(plot_data[plot_data.category==type_select].brand.unique())]

        st.markdown(""" 
                    * The left part shows the ranking of brands in the selected category
                    * Right part shows the average price of brands under the selected category 
                    \n
                     -----------------------------------
                    """)

        col1.write(list(data_req.groupby('brand').price_dollars.mean().sort_values(ascending=False).reset_index().brand))

        with col2:
            plost.bar_chart(
                np.round(data_req.groupby('brand').price_dollars.mean(),2).reset_index(),
                bar='brand',
                value='price_dollars',
                use_container_width=True,
                color='price_dollars'
            )

        st.markdown('##### :blue[Grade share by brand filtering]')
        st.markdown(""" 
                    * The values are in percentage [count]
                    * This is dataset representation not universal stats """)

        brand_select=st.selectbox('select brand for grade share',tuple(req_data.brand.unique()))

        plost.pie_chart(
            np.round(req_data[req_data.brand==brand_select]\
                .grade.value_counts()*100/req_data[req_data.brand==brand_select].shape[0],2).reset_index(),
            theta='count',
            color='grade',
            use_container_width=True
            )




    elif selection=='Numerical Data':

        plot_1=car_data[['price_dollars','year_of_manufacture','num_of_doors','engine_capacity']]

        for i in plot_1.columns:
            st.write(i)
            if i in 'price_dollars':
                plost.hist(
                    plot_1,
                    x=i,
                    bin=100,
                    use_container_width=True
                )
            
            else:
                plost.bar_chart(
                    plot_1[i].value_counts().reset_index(),
                    bar=i,
                    value='count',
                    use_container_width=True
                )

def comparision_tab():

    car_data=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/car_data.csv')

    price_ranges = [0, 30000, 60000, 100000, 200000, float('inf')]
    labels = ['Entry-Level', 'Mid-Range', 'Premium/Executive', 'Luxury/High-End', 'Ultra-Luxury/Exotic']

    car_data['coded_brand']=pd\
            .cut(car_data.price_dollars,bins=price_ranges,labels=labels)
        
    labels_encoded={'Entry-Level':1, 'Mid-Range':2, 'Premium/Executive':3, 'Luxury/High-End':4, 'Ultra-Luxury/Exotic':5}
    car_data['labels_encoded']=car_data.coded_brand.map(labels_encoded)
    car_data['labels_encoded']=car_data.labels_encoded.astype('int64')


    compare_select=st.sidebar.selectbox('Select type of comparision:',['All data comparision','Brand to Brand comparision','Model to Model comparision'],index=1)

    if compare_select=='All data comparision':
        
        col1,col2=st.columns(2)

        st.subheader('Numeric Column Visualization:')
        numeric_columns = car_data[['mileage', 'num_of_doors', 'seating_capacity', 'year_of_manufacture', 'price_dollars', 'fuel_per_100km']].columns
        selected_numeric_column = st.selectbox('Select a numeric column:', numeric_columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(car_data[selected_numeric_column], kde=True, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {selected_numeric_column}', fontsize=16)
        ax.set_xlabel(selected_numeric_column.capitalize(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)

        st.subheader('Categorical Column Visualization:')
        categorical_columns = car_data[['condition', 'car_model', 'exterior_color', 'interior_color',
                                        'transmission', 'drive_type', 'brand', 'grade', 'engine_type']].columns
        selected_categorical_column = st.selectbox('Select a categorical column:', categorical_columns)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=car_data, x=selected_categorical_column, palette='pastel', ax=ax)
        ax.set_title(f'Count of {selected_categorical_column.capitalize()}', fontsize=16)
        ax.set_xlabel(selected_categorical_column.capitalize(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader('Relationship between Numeric and Categorical Columns:')
        selected_x_numeric = st.selectbox('Select numeric column for x-axis:', numeric_columns)
        selected_y_numeric = st.selectbox('Select numeric column for y-axis:', numeric_columns)
        selected_hue_categorical = st.selectbox('Select categorical column for hue:', categorical_columns)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=car_data, x=selected_x_numeric, y=selected_y_numeric, hue=selected_hue_categorical, palette='viridis', ax=ax)
        ax.set_title(f'Scatter Plot - {selected_x_numeric} vs {selected_y_numeric} with {selected_hue_categorical}', fontsize=16)
        ax.set_xlabel(selected_x_numeric.capitalize(), fontsize=12)
        ax.set_ylabel(selected_y_numeric.capitalize(), fontsize=12)
        st.pyplot(fig)

        st.subheader('Box Plot for Numeric and Categorical Columns:')
        selected_box_numeric = st.selectbox('Select numeric column for box plot:', numeric_columns)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x=selected_categorical_column, y=selected_box_numeric, data=car_data, palette='Set2', ax=ax)
        ax.set_title(f'Box Plot - {selected_categorical_column} vs {selected_box_numeric}', fontsize=16)
        ax.set_xlabel(selected_categorical_column.capitalize(), fontsize=12)
        ax.set_ylabel(selected_box_numeric.capitalize(), fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader('Violin Plot for Numeric and Categorical Columns:')
        selected_violin_numeric = st.selectbox('Select numeric column for violin plot:', numeric_columns)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(x=selected_categorical_column, y=selected_violin_numeric, data=car_data, palette='pastel', ax=ax)
        ax.set_title(f'Violin Plot - {selected_categorical_column} vs {selected_violin_numeric}', fontsize=16)
        ax.set_xlabel(selected_categorical_column.capitalize(), fontsize=12)
        ax.set_ylabel(selected_violin_numeric.capitalize(), fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader('Pairplot for Multiple Numeric Columns:')
        selected_pairplot_columns = st.multiselect('Select numeric columns for pairplot:', numeric_columns)

        if selected_pairplot_columns:
            fig = sns.pairplot(car_data[selected_pairplot_columns])
            st.pyplot(fig)

    elif compare_select=='Brand to Brand comparision':
        
        col1,col2=st.columns((5,5))

        with col1:
            brand_1=st.selectbox('Select First Brand',car_data.brand.unique(),index=1)

            plot_1=car_data[car_data.brand==brand_1]

            st.write(f'{brand_1} has an average cost rating of &nbsp; : &nbsp; {np.round(plot_1.labels_encoded.mean(),1)}')
            st.write(f'It is graded as an :blue[{plot_1.coded_brand.agg(pd.Series.mode)[0]}] Car')
            

            st.markdown('-----------')

            plost.line_chart(
                plot_1[plot_1.year_of_manufacture>2000]\
                    .groupby('year_of_manufacture').price_dollars.mean().reset_index(),
                x='year_of_manufacture',
                title=f'{brand_1} price change across years',
                y='price_dollars',
                use_container_width=True
            )


            st.markdown('---------')

            plost.bar_chart(
                np.round(plot_1.groupby('grade').price_dollars.mean(),2).reset_index(),
                bar='grade',
                value='price_dollars',
                use_container_width=True,
                color='grade',
                title=f'price distribution across models in  {brand_1}'
                )
            
            st.markdown('--------------')

            st.markdown(f'##### :blue[Car models under each type for {brand_1}]')

            st.write(dict(plot_1.groupby('car_model').grade.nunique()))


            

        with col2:
            
            brand_2=st.selectbox('Select Second Brand',car_data.brand.unique(),index=2)
            plot_2=car_data[car_data.brand==brand_2]

            st.write(f'{brand_2} has an average cost rating of &nbsp; : &nbsp; {np.round(plot_2.labels_encoded.mean(),1)}')
            st.write(f'It is graded as an :red[{plot_2.coded_brand.agg(pd.Series.mode)[0]}] Car')
            
            st.markdown('-----------')

            plost.line_chart(
                plot_2[plot_2.year_of_manufacture>2000]\
                    .groupby('year_of_manufacture').price_dollars.mean().reset_index(),
                x='year_of_manufacture',
                title=f'{brand_2} price change across years',
                y='price_dollars',
                use_container_width=True
            )

            st.markdown('---------')

            plost.bar_chart(
                np.round(plot_2.groupby('grade').price_dollars.mean(),2).reset_index(),
                bar='grade',
                value='price_dollars',
                use_container_width=True,
                color='grade',
                title=f'Price distribution models in  {brand_2}'
            )

            st.markdown('------------------------------------')

            st.markdown(f'##### :red[Car models under each type for {brand_2}]')

            st.write(dict(plot_2.groupby('car_model').grade.nunique()))

        st.markdown('----------------------------')

        col11,col22=st.columns((5,5))

        with col11:

            plost.bar_chart(
                np.round(plot_1.groupby('car_model').price_dollars.mean(),2).reset_index(),
                bar='car_model',
                value='price_dollars',
                color='price_dollars',
                title=f'Average price per car type - {brand_1}',
                use_container_width=True
            )

            st.markdown('-----------------')

            plost.pie_chart(
                np.round(plot_1['drive_type'].value_counts()*100/plot_1.shape[0],2).reset_index(),
                theta='count',
                color='drive_type',
                use_container_width=True,
                title=f'Drive Type distribution for {brand_1}'
            )

            st.markdown('----------')

            plost.pie_chart(
                np.round(plot_1['engine_type'].value_counts()*100/plot_1.shape[0],2).reset_index(),
                theta='count',
                color='engine_type',
                use_container_width=True,
                height=200,
                title=f'Engine type distribution for {brand_1}'
            )

            st.markdown('---------------')

            plost.bar_chart(
                plot_1.exterior_color.value_counts().reset_index(),
                title=f'exterior color distribution for {brand_1}',
                bar='exterior_color',
                value='count',
                use_container_width=True,
                color='exterior_color'
            )

        with col22:

            plost.bar_chart(
                np.round(plot_2.groupby('car_model').price_dollars.mean(),2).reset_index(),
                bar='car_model',
                value='price_dollars',
                color='price_dollars',
                title=f'Average price per car type - {brand_2}',
                use_container_width=True
            )

            st.markdown('---------------')
            
            plost.pie_chart(
                np.round(plot_2['drive_type'].value_counts()*100/plot_2.shape[0],2).reset_index(),
                theta='count',
                color='drive_type',
                use_container_width=True,
                title=f'Drive Type distribution for {brand_2}'
            )

            st.markdown('----------')

            plost.pie_chart(
                np.round(plot_2['engine_type'].value_counts()*100/plot_2.shape[0],2).reset_index(),
                theta='count',
                color='engine_type',
                use_container_width=True,
                height=200,
                title=f'Engine type distribution for {brand_2}'
            )

            st.markdown('---------------')

            plost.bar_chart(
                plot_2.exterior_color.value_counts().reset_index(),
                title=f'exterior color distribution for {brand_2}',
                bar='exterior_color',
                value='count',
                use_container_width=True,
                color='exterior_color'
            )

    elif compare_select=='Model to Model comparision':
        other_page()

def download_page():

    car_data=pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/car_data.csv')
    st.title('Car Data Visualization App')
    car_data.drop(columns=['Unnamed: 0','ad_id'],inplace=True)
    price_ranges = [0, 30000, 60000, 100000, 200000, float('inf')]
    labels = ['Entry-Level', 'Mid-Range', 'Premium/Executive', 'Luxury/High-End', 'Ultra-Luxury/Exotic']

    car_data['coded_brand']=pd\
            .cut(car_data.price_dollars,bins=price_ranges,labels=labels)
        
    labels_encoded={'Entry-Level':1, 'Mid-Range':2, 'Premium/Executive':3, 'Luxury/High-End':4, 'Ultra-Luxury/Exotic':5}
    car_data['labels_encoded']=car_data.coded_brand.map(labels_encoded)
    car_data['labels_encoded']=car_data.labels_encoded.astype('int64')


    st.subheader('Sample Data:')
    st.dataframe(car_data.head())

    st.title('Filter and Download Data')

    st.sidebar.subheader('Filter Options:')
    
    condition_filter = st.sidebar.selectbox('Select Condition:', car_data['condition'].unique(),index=None)

    year_range = st.sidebar.slider('Select Year Range:', min_value=car_data['year_of_manufacture'].min(), max_value=car_data['year_of_manufacture'].max(), value=(car_data['year_of_manufacture'].min(), car_data['year_of_manufacture'].max()))

    price_category_filter = st.sidebar.selectbox('Select Price Category:', labels,index=None)
    
    if price_category_filter:
        brand_filter=st.sidebar.selectbox('Select Brand :',car_data[car_data.coded_brand==price_category_filter].brand.unique(),index=None)
        if brand_filter:
            grade_filter=st.sidebar.selectbox('Select Model',car_data[(car_data.brand==brand_filter)&(car_data.coded_brand==price_category_filter)].grade.unique(),index=None)
    
    else:
        brand_filter=st.sidebar.selectbox('Select Brand :',car_data.brand.unique(),index=None)
        if brand_filter:
            grade_filter=st.sidebar.selectbox('Select Model',car_data[(car_data.brand==brand_filter)].unique(),index=None)
        

    

    transmission_filter = st.sidebar.selectbox('Select Transmission:', car_data['transmission'].unique(),index=None)

    price_range = st.sidebar.slider('Select Price Range:', min_value=car_data['price_dollars'].min(), max_value=car_data['price_dollars'].max(), value=(car_data['price_dollars'].min(), car_data['price_dollars'].max()))

    


    filtered_data = car_data[
        (car_data['year_of_manufacture'].between(year_range[0], year_range[1])) &
        (car_data['price_dollars'].between(price_range[0], price_range[1]))
    ]

    if price_category_filter:
        filtered_data=filtered_data[filtered_data['coded_brand'] == price_category_filter]

    if condition_filter:
        filtered_data=filtered_data[filtered_data['condition'] == condition_filter]

    if transmission_filter:
        filtered_data=filtered_data[filtered_data['transmission'] == transmission_filter]

    if brand_filter:
        filtered_data=filtered_data[filtered_data.brand==brand_filter]
        if grade_filter:
            filtered_data=filtered_data[filtered_data.grade==grade_filter]

    

    
    filtered_data=filtered_data.reset_index(drop=True)
    filtered_data.drop(columns=['brand_code','labels_encoded','age','zscore','brand_grade','extracted_name',
                                'price_log','fuel_per_100km','engine_capacity','engine_type','changed_price',
                                'car_price','price','coded_brand'],inplace=True)

    st.subheader('Filtered Data:')
    st.dataframe(filtered_data)
    st.write(f'Data has {filtered_data.shape[0]} rows')

    st.subheader('Download Filtered Dataset:')
    csv_file_filtered = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered CSV",
        data=csv_file_filtered,
        file_name="filtered_car_data.csv",
        mime="text/csv",
    )

def other_page():

    # Load car_data from CSV file
    car_data = pd.read_csv('/Users/dhruvparikh/Downloads/Untitled Folder/car_data.csv')
    st.title('Brand and Model Comparison Page')

    # Set Seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid")

    # Sidebar for brand and model selection
    st.sidebar.subheader('Select Brands:')
    selected_brand1 = st.sidebar.selectbox('Select Brand 1:', car_data['brand'].unique(), index=0)
    selected_brand2 = st.sidebar.selectbox('Select Brand 2:', car_data['brand'].unique(), index=1)

    # Filter grades based on selected brands
    grade_options_brand1 = car_data[car_data['brand'] == selected_brand1]['grade'].unique()
    grade_options_brand2 = car_data[car_data['brand'] == selected_brand2]['grade'].unique()

    st.sidebar.subheader('Select Model:')
    selected_grade_brand1 = st.sidebar.selectbox(f'Select {selected_brand1} Model:', grade_options_brand1, index=0)
    selected_grade_brand2 = st.sidebar.selectbox(f'Select {selected_brand2} Model:', grade_options_brand2, index=0)

    # Filter data for selected brands and grade
    filtered_data_brand1_grade = car_data[(car_data['brand'] == selected_brand1) & (car_data['grade'] == selected_grade_brand1)]
    filtered_data_brand2_grade = car_data[(car_data['brand'] == selected_brand2) & (car_data['grade'] == selected_grade_brand2)]

    st.subheader('Brand and Grade Comparison Visualizations:')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.barplot(x='brand', y='price_dollars', data=filtered_data_brand1_grade, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title(f'Average Price for {selected_brand1} - {selected_grade_brand1}')

    sns.barplot(x='brand', y='price_dollars', data=filtered_data_brand2_grade, ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title(f'Average Price for {selected_brand2} - {selected_grade_brand2}')

    sns.countplot(x='transmission', data=filtered_data_brand1_grade, ax=axes[1, 0], palette="Set3")
    axes[1, 0].set_title(f'Transmission Types for {selected_brand1} - {selected_grade_brand1}')

    sns.countplot(x='transmission', data=filtered_data_brand2_grade, ax=axes[1, 1], palette="Set3")
    axes[1, 1].set_title(f'Transmission Types for {selected_brand2} - {selected_grade_brand2}')

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader('Violin Plots for Numerical Features:')

    additional_columns = ['origin', 'condition', 'car_model', 'mileage', 'exterior_color',
                        'interior_color', 'seating_capacity', 'transmission', 'drive_type',
                        'engine_type', 'engine_capacity', 'age']

    for column in additional_columns:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.violinplot(x='brand', y=column, data=filtered_data_brand1_grade, palette="viridis", inner="quartile")
        plt.title(f'{column} Comparison for {selected_brand1} - {selected_grade_brand1}')

        plt.subplot(1, 2, 2)
        sns.violinplot(x='brand', y=column, data=filtered_data_brand2_grade, palette="viridis", inner="quartile")
        plt.title(f'{column} Comparison for {selected_brand2} - {selected_grade_brand2}')

        plt.tight_layout()

        st.pyplot()

    st.subheader('Swarm and Strip Plots for Numerical Features:')

    for column in additional_columns:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.swarmplot(x='brand', y=column, data=filtered_data_brand1_grade, palette="viridis", size=4)
        plt.title(f'{column} Comparison for {selected_brand1} - {selected_grade_brand1}')

        plt.subplot(1, 2, 2)
        sns.stripplot(x='brand', y=column, data=filtered_data_brand2_grade, palette="viridis", size=4, jitter=True)
        plt.title(f'{column} Comparison for {selected_brand2} - {selected_grade_brand2}')

        plt.tight_layout()

        st.pyplot()


if __name__ == "__main__":
    main()
