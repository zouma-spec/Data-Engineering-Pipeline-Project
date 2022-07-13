import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

st.write(
'''
## NYC Property Appraisal
''')

pickle_in = open('/Users/Layla/Desktop/Metis/Engineering/project/lm.pkl', 'rb')
lm_model = pickle.load(pickle_in)

def predictor():
    input_data=([[TOTAL_UNITS,GROSS_SQUARE_FEET,YEAR_BUILT,units_square_feet,BOROUGH,NEIGHBORHOOD,BUILDING_CLASS_CATEGORY]])
    prediction = lm_model.predict(input_data)
    
    return int(prediction)

# Get input values - numeric variables
TOTAL_UNITS = st.slider('Please enter the number of units:',
                                 min_value = 0,
                                 max_value = 2261
                                )

GROSS_SQUARE_FEET = st.slider('Please enter the gross square feet:',
                                 min_value = 0 ,
                                 max_value = 2100000
                                )
YEAR_BUILT = st.slider('Please enter the year the property built:',
                                 min_value = 1800,
                                 max_value = 2022
                                )

units_square_feet=TOTAL_UNITS*GROSS_SQUARE_FEET

BOROUGH=st.selectbox('Please choose the borough:',
                              ('Manhattan',
                               'Bronx',
                               'Brooklyn',
                               'Queens', 
                               'Staten_Island'
                               )
                             )


NEIGHBORHOOD=st.selectbox('Please choose the neighborhood:',
                              ('BATHGATE', 'BAYCHESTER', 'BEDFORD PARK/NORWOOD', 'BELMONT',
       'BRONX PARK', 'BRONXDALE', 'CASTLE HILL/UNIONPORT', 'CITY ISLAND',
       'CITY ISLAND-PELHAM STRIP', 'COUNTRY CLUB', 'CROTONA PARK',
       'EAST TREMONT', 'FIELDSTON', 'FORDHAM',
       'HIGHBRIDGE/MORRIS HEIGHTS', 'HUNTS POINT',
       'KINGSBRIDGE HTS/UNIV HTS', 'KINGSBRIDGE/JEROME PARK',
       'MELROSE/CONCOURSE', 'MORRIS PARK/VAN NEST', 'MORRISANIA/LONGWOOD',
       'MOTT HAVEN/PORT MORRIS', 'MOUNT HOPE/MOUNT EDEN', 'PARKCHESTER',
       'PELHAM GARDENS', 'PELHAM PARKWAY NORTH', 'PELHAM PARKWAY SOUTH',
       'RIVERDALE', 'SCHUYLERVILLE/PELHAM BAY', 'SOUNDVIEW',
       'THROGS NECK', 'WAKEFIELD', 'WESTCHESTER', 'WILLIAMSBRIDGE',
       'WOODLAWN', 'BATH BEACH', 'BAY RIDGE', 'BEDFORD STUYVESANT',
       'BENSONHURST', 'BERGEN BEACH', 'BOERUM HILL', 'BOROUGH PARK',
       'BRIGHTON BEACH', 'BROOKLYN HEIGHTS', 'BROWNSVILLE',
       'BUSH TERMINAL', 'BUSHWICK', 'CANARSIE', 'CARROLL GARDENS',
       'CLINTON HILL', 'COBBLE HILL', 'COBBLE HILL-WEST', 'CONEY ISLAND',
       'CROWN HEIGHTS', 'CYPRESS HILLS', 'DOWNTOWN-FULTON MALL',
       'DYKER HEIGHTS', 'EAST NEW YORK', 'FLATBUSH-CENTRAL',
       'FLATBUSH-EAST', 'FLATBUSH-LEFFERTS GARDEN', 'FLATBUSH-NORTH',
       'FLATLANDS', 'FORT GREENE', 'GERRITSEN BEACH', 'GOWANUS',
       'GRAVESEND', 'GREENPOINT', 'KENSINGTON', 'MADISON',
       'MANHATTAN BEACH', 'MARINE PARK', 'MIDWOOD', 'MILL BASIN',
       'NAVY YARD', 'OCEAN HILL', 'OCEAN PARKWAY-NORTH',
       'OCEAN PARKWAY-SOUTH', 'OLD MILL BASIN', 'PARK SLOPE',
       'PARK SLOPE SOUTH', 'PROSPECT HEIGHTS', 'RED HOOK', 'SEAGATE',
       'SHEEPSHEAD BAY', 'SPRING CREEK', 'SUNSET PARK',
       'WILLIAMSBURG-CENTRAL', 'WILLIAMSBURG-EAST', 'WILLIAMSBURG-NORTH',
       'WILLIAMSBURG-SOUTH', 'WINDSOR TERRACE', 'WYCKOFF HEIGHTS',
       'ALPHABET CITY', 'CHELSEA', 'CLINTON', 'EAST VILLAGE', 'FASHION',
       'FINANCIAL', 'FLATIRON', 'GRAMERCY', 'GREENWICH VILLAGE-WEST',
       'HARLEM-CENTRAL', 'HARLEM-EAST', 'HARLEM-UPPER', 'HARLEM-WEST',
       'INWOOD', 'KIPS BAY', 'LITTLE ITALY', 'LOWER EAST SIDE',
       'MANHATTAN VALLEY', 'MIDTOWN CBD', 'MIDTOWN EAST',
       'MORNINGSIDE HEIGHTS', 'MURRAY HILL', 'SOHO', 'SOUTHBRIDGE',
       'TRIBECA', 'UPPER EAST SIDE (59-79)', 'UPPER EAST SIDE (79-96)',
       'UPPER EAST SIDE (96-110)', 'UPPER WEST SIDE (59-79)',
       'UPPER WEST SIDE (79-96)', 'UPPER WEST SIDE (96-116)',
       'WASHINGTON HEIGHTS LOWER', 'WASHINGTON HEIGHTS UPPER',
       'AIRPORT LA GUARDIA', 'ARVERNE', 'ASTORIA', 'BAYSIDE',
       'BEECHHURST', 'BELLE HARBOR', 'BELLEROSE', 'BRIARWOOD',
       'BROAD CHANNEL', 'CAMBRIA HEIGHTS', 'COLLEGE POINT', 'CORONA',
       'DOUGLASTON', 'EAST ELMHURST', 'ELMHURST', 'FAR ROCKAWAY',
       'FLORAL PARK', 'FLUSHING-NORTH', 'FLUSHING-SOUTH', 'FOREST HILLS',
       'FRESH MEADOWS', 'GLEN OAKS', 'GLENDALE', 'HAMMELS', 'HILLCREST',
       'HOLLIS', 'HOLLIS HILLS', 'HOLLISWOOD', 'HOWARD BEACH',
       'JACKSON HEIGHTS', 'JAMAICA', 'JAMAICA BAY', 'JAMAICA ESTATES',
       'JAMAICA HILLS', 'KEW GARDENS', 'LAURELTON', 'LITTLE NECK',
       'LONG ISLAND CITY', 'MASPETH', 'MIDDLE VILLAGE', 'NEPONSIT',
       'OAKLAND GARDENS', 'OZONE PARK', 'QUEENS VILLAGE', 'REGO PARK',
       'RICHMOND HILL', 'RIDGEWOOD', 'ROCKAWAY PARK', 'ROSEDALE',
       'SO. JAMAICA-BAISLEY PARK', 'SOUTH JAMAICA', 'SOUTH OZONE PARK',
       'SPRINGFIELD GARDENS', 'ST. ALBANS', 'SUNNYSIDE', 'WHITESTONE',
       'WOODHAVEN', 'WOODSIDE', 'ANNADALE', 'ARDEN HEIGHTS', 'ARROCHAR',
       'ARROCHAR-SHORE ACRES', 'BLOOMFIELD', 'BULLS HEAD',
       'CASTLETON CORNERS', 'CLOVE LAKES', 'CONCORD', 'CONCORD-FOX HILLS',
       'DONGAN HILLS', 'DONGAN HILLS-COLONY', 'DONGAN HILLS-OLD TOWN',
       'ELTINGVILLE', 'EMERSON HILL', 'FRESH KILLS', 'GRANT CITY',
       'GRASMERE', 'GREAT KILLS', 'GREAT KILLS-BAY TERRACE',
       'GRYMES HILL', 'HUGUENOT', 'LIVINGSTON', 'MANOR HEIGHTS',
       'MARINERS HARBOR', 'MIDLAND BEACH', 'NEW BRIGHTON',
       'NEW BRIGHTON-ST. GEORGE', 'NEW DORP', 'NEW DORP-BEACH',
       'NEW DORP-HEIGHTS', 'NEW SPRINGVILLE', 'OAKWOOD', 'OAKWOOD-BEACH',
       'PLEASANT PLAINS', 'PORT IVORY', 'PORT RICHMOND', 'PRINCES BAY',
       'RICHMONDTOWN', 'RICHMONDTOWN-LIGHTHS HILL', 'ROSEBANK',
       'ROSSVILLE', 'ROSSVILLE-CHARLESTON', 'ROSSVILLE-RICHMOND VALLEY',
       'SILVER LAKE', 'SOUTH BEACH', 'STAPLETON', 'STAPLETON-CLIFTON',
       'TODT HILL', 'TOMPKINSVILLE', 'TOTTENVILLE', 'TRAVIS',
       'WEST NEW BRIGHTON', 'WESTERLEIGH', 'WILLOWBROOK', 'WOODROW'
                               )
                             )

BUILDING_CLASS_CATEGORY=st.selectbox('Please choose the building class category:',
    ('01 ONE FAMILY DWELLINGS', '02 TWO FAMILY DWELLINGS',
       '03 THREE FAMILY DWELLINGS', '22 STORE BUILDINGS',
       '29 COMMERCIAL GARAGES', '30 WAREHOUSES',
       '37 RELIGIOUS FACILITIES', '06 TAX CLASS 1 - OTHER',
       '07 RENTALS - WALKUP APARTMENTS', '14 RENTALS - 4-10 UNIT',
       '21 OFFICE BUILDINGS', '38 ASYLUMS AND HOMES',
       '33 EDUCATIONAL FACILITIES', '08 RENTALS - ELEVATOR APARTMENTS',
       '27 FACTORIES', '41 TAX CLASS 4 - OTHER',
       '35 INDOOR PUBLIC AND CULTURAL FACILITIES',
       '05 TAX CLASS 1 VACANT LAND', '11 SPECIAL CONDO BILLING LOTS',
       '25 LUXURY HOTELS', '32 HOSPITAL AND HEALTH FACILITIES'
        )
    )


# Making predictions    


df = pd.read_csv('/Users/Layla/Desktop/Metis/Engineering/project/x_train.csv')
df=df.drop('Unnamed: 0',1)        
column_names=df.columns
arr=np.zeros(267)
arr = arr.reshape(1,267)
input_data=pd.DataFrame(arr,columns=column_names)

input_data.loc[:,'TOTAL_UNITS']=TOTAL_UNITS
input_data.loc[:,'GROSS_SQUARE_FEET']=GROSS_SQUARE_FEET
input_data.loc[:,'YEAR_BUILT']=YEAR_BUILT
input_data.loc[:,'units_square_feet']=units_square_feet


BOROUGH = 'BOROUGH_'+ BOROUGH 
if BOROUGH== 'BOROUGH_Bronx': 
	pass
else: 
	input_data.loc[:,BOROUGH]=1


NEIGHBORHOOD = 'NEIGHBORHOOD_' + NEIGHBORHOOD
if NEIGHBORHOOD == 'NEIGHBORHOOD_AIRPORT LA GUARDIA':
    pass
else:
    input_data.loc[:,NEIGHBORHOOD]=1



BUILDING_CLASS_CATEGORY= 'BUILDING_CLASS_CATEGORY_' + BUILDING_CLASS_CATEGORY
if BUILDING_CLASS_CATEGORY =='BUILDING_CLASS_CATEGORY_00141 TAX CLASS 4 - OTHER':
    pass
else:
    input_data.loc[:,BUILDING_CLASS_CATEGORY]=1


#result =lm_model.predict([[TOTAL_UNITS,GROSS_SQUARE_FEET,YEAR_BUILT,units_square_feet,BOROUGH,NEIGHBORHOOD,BUILDING_CLASS_CATEGORY]])
result=lm_model.predict(input_data)
result = np.round(result)
st.success('The property worths around $ {}'.format(result))

