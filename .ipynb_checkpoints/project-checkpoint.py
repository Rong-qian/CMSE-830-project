import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('Project on MPG')

st.header('Introduction')

st.markdown(
    'In this project, I will explore two datasets on car to provide suggestion for custom wants to buy a used car. \
    They are mpg dataset :https://www.kaggle.com/datasets/uciml/autompg-dataset and 100,000 UK Used Car Data set:https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?resource=download.\
    The goal of the project is to explore how to find a car with better fuel efficiency and find the tendency of the fuel efficiency with the development of car industry '
)

st.header('MPG dataset')

########################################
########### Data reading ###############
########################################
## Load MPG dataset
df_mpg = sns.load_dataset("mpg")

st.subheader('Overview')

########################################
########### Overview     ###############
########################################
st.markdown('First thing to check is what are in the MPG dataset.\
            In MPG dataset, there is origin to show where the car comes from and the name of the model\
            Others are some features related to the mpg. Use the selectbox to view the relation between different features\
            If two features are the same, the histogram and violin plot will be shown instead.')
# Example options for the selectbox
options = ['mpg', 'cylinders', 'displacement', 'horsepower','weight','acceleration','model_year']

selected_option_x = st.selectbox('Choose first feature:', options)
selected_option_y = st.selectbox('Choose second feature:', options)

# Display the selected option
alt.renderers.enable('mimetype')
if selected_option_x != selected_option_y:
    chart = alt.Chart(df_mpg).mark_point().encode(
        x=selected_option_x,
        y=selected_option_y,
        color='origin'
    ).properties(
        width=500,
        height=400
    ).interactive()
    st.altair_chart(chart)

else:
    chart_hist = alt.Chart(df_mpg).mark_bar(opacity=0.3).encode(
        alt.X(selected_option_x +":Q", bin=True),
        alt.Y('count()').stack(None),
        color='origin'
    ).properties(
        width=350,
        height=300
    ).interactive()

    chart_violin = alt.Chart(df_mpg, width=100).transform_density(
        selected_option_x,
        as_=[selected_option_x, 'density'],
        groupby=['origin']
    ).mark_area(orient='horizontal').encode(
        alt.X('density:Q')
            .stack('center')
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0], grid=False, ticks=True),
        alt.Y(str(selected_option_x)+':Q'),
        alt.Color('origin:N'),
        alt.Column('origin:N')
            .spacing(0)
            .header(titleOrient='bottom', labelOrient='bottom', labelPadding=0)
    ).properties(
        width=100,
        height=300
    ).interactive()

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_hist)
    with col2:
        st.altair_chart(chart_violin)

########################################
############### Draw missing  ##########
########################################

st.subheader('Missing value')

st.markdown('In MPG dataset, there is origin to show where the car comes from and the name of the model.\
             Others are some features related to the mpg. I first check the missing data. \
             and find the horsepower column has some missing value.')

df_mpg_subset = df_mpg[["mpg", "cylinders", "displacement", "horsepower","weight","acceleration","model_year"]]
nan_mask = df_mpg_subset.isna()
nan_array = nan_mask.astype(int).to_numpy()
plt.figure(figsize=(12, 6))
im = plt.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')

plt.xlabel('data Index')
plt.ylabel('Features')
plt.title('Visualizing Missing Values in mpg')
plt.yticks(range(len(df_mpg_subset.columns)), df_mpg_subset.columns)

num_planets = nan_array.shape[0]


plt.xticks(np.linspace(0, num_planets-1, min(10, num_planets)).astype(int))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

st.pyplot(plt)


########################################
############# Imputation ###############
########################################

st.subheader('Missing value')


numeric_columns = df_mpg.select_dtypes(include=[np.number]).columns
df_mpg_numeric = df_mpg[numeric_columns]
string_columns = df_mpg.select_dtypes(include=[object]).columns
df_mpg_string = df_mpg[string_columns]


st.markdown('By the missing heatmap, the horsepower column has few missing value.\
            I will use stochastic regression method to do the imputation. \
            Before implementing the imputation, heatmap of correlation may help determine which variable is useful \
            It shows the car specifications like weight and acceleration has negatives correlation with mpg. \
            Also the mpg have positive correlation with year, which show cars with better fuel economy are more popular in market.'
            )
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(df_mpg_numeric.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt)
st.markdown('Variables expect for acceleartion and model year all have negative correlation with MPG. \
            I will use those features for imputation.\
            Next two plots show the stochastic regression result for test and missing data.')

df_mpg_clean = df_mpg.dropna()

# define predictors and target
X = df_mpg_clean[['mpg', 'weight','cylinders','displacement']]
y = df_mpg_clean['horsepower']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Calculate residuals from the training set
residuals = y_train - linear_model.predict(X_train)

# Estimate the standard deviation of the residuals
residual_std = np.std(residuals)

# Generate predictions using the original linear model
y_pred = linear_model.predict(X_test)

# Number of stochastic simulations
n_simulations = 100
stochastic_predictions = []

for i in range(n_simulations):
    # Add noise to the final predictions (assuming noise in the outcome)
    noise = np.random.normal(0, residual_std, size=y_pred.shape)  # residual_std from previous calculation
    stochastic_predictions.append(y_pred + noise)

# Convert to numpy array
stochastic_predictions = np.array(stochastic_predictions)

# predict on the test set
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')

y_pred = linear_model.predict(X_test)
plt.fill_between(range(len(y_test)), stochastic_predictions.min(axis=0), stochastic_predictions.max(axis=0),
                 color='blue', alpha=0.2, label='Stochastic Prediction Range')
plt.legend()
plt.title('Stochastic Regression for test: horsepower Predictions')
plt.xlabel('Sample Index')
plt.ylabel('horsepower')
st.pyplot(plt)

########################
# Now, do the same for missing
df_mpg_subset = df_mpg[["mpg", "cylinders", "displacement", "horsepower","weight","acceleration","model_year"]]

df_mpg_missing = df_mpg[df_mpg['horsepower'].isna()] 
predictions = linear_model.predict(df_mpg_missing[['mpg', 'weight','cylinders','displacement']])

# Fill the missing values in the original DataFrame
df_mpg.loc[df_mpg['horsepower'].isna(), 'horsepower'] = predictions

# Number of stochastic simulations
n_simulations = 100
stochastic_predictions = []

for i in range(n_simulations):
    # Add noise to the final predictions (assuming noise in the outcome)
    noise = np.random.normal(0, residual_std, size=predictions.shape)  # residual_std from previous calculation
    stochastic_predictions.append(predictions + noise)

# Convert to numpy array
predictions = np.array(predictions)

# predict on the test set
plt.figure(figsize=(10, 6))
plt.scatter(range(len(predictions)), predictions, color='black', label='Actual')
stochastic_predictions = np.array(stochastic_predictions)

plt.fill_between(range(len(predictions)), stochastic_predictions.min(axis=0), stochastic_predictions.max(axis=0),
                 color='blue', alpha=0.2, label='Stochastic Prediction Range')
plt.legend()
plt.title('Stochastic Regression for missing values: horsepower Predictions')
plt.xlabel('Sample Index')
plt.ylabel('horsepower')
st.pyplot(plt)



########################################
############## Scale ###################
########################################

st.subheader('Scale')

st.markdown('Use the standard scale to handle the data.Then use the selectbox to see the difference before/after scaling')


numeric_columns = df_mpg.select_dtypes(include=[np.number]).columns
df_mpg_numeric = df_mpg[numeric_columns]
string_columns = df_mpg.select_dtypes(include=[object]).columns
df_mpg_string = df_mpg[string_columns]

scaler = StandardScaler() #Scale the data:standardize features by removing the mean and scaling to unit variance.

df_mpg_numeric_scaled = scaler.fit_transform(df_mpg_numeric) # Get Scaled  data
df_mpg_numeric_scaled = pd.DataFrame(df_mpg_numeric_scaled, columns=df_mpg_numeric.columns)

df_mpg_scaled = pd.concat([df_mpg_numeric_scaled, df_mpg_string], axis=1)


# Example options for the selectbox
options = ['mpg', 'cylinders', 'displacement', 'horsepower','weight','acceleration']

selected_option_x_scale = st.selectbox('See the scale: Choose first feature:', options)
selected_option_y_scale = st.selectbox('See the scale: Choose second feature :', options)

# Display the selected option
alt.renderers.enable('mimetype')
if selected_option_x_scale != selected_option_y_scale:
    chart_noscale = alt.Chart(df_mpg).mark_point().encode(
        x=selected_option_x_scale,
        y=selected_option_y_scale,
        color='origin'
    ).properties(
        width=300,
        height=200
    ).interactive()

    chart_scaled = alt.Chart(df_mpg_scaled).mark_point().encode(
        x=selected_option_x_scale,
        y=selected_option_y_scale,
        color='origin'
    ).properties(
        width=300,
        height=200
    ).interactive()

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_noscale)
    with col2:
        st.altair_chart(chart_scaled)

else:
    chart_hist_noscale = alt.Chart(df_mpg).mark_bar(opacity=0.3).encode(
        alt.X(selected_option_x_scale +":Q", bin=True),
        alt.Y('count()').stack(None),
        color='origin'
    ).properties(
        width=300,
        height=200
    ).interactive()

    chart_hist_scaled = alt.Chart(df_mpg_scaled).mark_bar(opacity=0.3).encode(
        alt.X(selected_option_x_scale +":Q", bin=True),
        alt.Y('count()').stack(None),
        color='origin'
    ).properties(
        width=300,
        height=200
    ).interactive()

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_hist_noscale)
    with col2:
        st.altair_chart(chart_hist_scaled)
plt.figure(figsize=(10, 6))

df_mpg_japan = df_mpg[df_mpg["origin"]=="japan"]
df_mpg_usa = df_mpg[df_mpg["origin"]=="usa"]

model_japan = LinearRegression()
model_japan.fit(np.array(df_mpg_japan['horsepower']).reshape(-1, 1), np.array(df_mpg_japan['mpg']).reshape(-1, 1))  
X_japan = np.linspace(df_mpg_japan['horsepower'].min(), df_mpg_japan['horsepower'].max(), 100).reshape(-1, 1)
y_japan = model_japan.predict(X_japan)
plt.scatter(df_mpg_japan['horsepower'], df_mpg_japan['mpg'], color='blue', label='Data points')
plt.plot(X_japan, y_japan, color='red', label='Regression line')

model_usa = LinearRegression()
model_usa.fit(np.array(df_mpg_usa['horsepower']).reshape(-1, 1), np.array(df_mpg_usa['mpg']).reshape(-1, 1))  
X_usa = np.linspace(df_mpg_usa['horsepower'].min(), df_mpg_usa['horsepower'].max(), 100).reshape(-1, 1)
y_usa = model_usa.predict(X_usa)
plt.scatter(df_mpg_usa['horsepower'], df_mpg_usa['mpg'], color='green', label='USA Data points')
plt.plot(X_usa, y_usa, color='black', label='USA Regression line')
plt.legend()
plt.xlabel('horsepower')
plt.ylabel('mpg')
st.pyplot(plt)
#######################################
########### Narrative     ###############
########################################
st.subheader('Narrative')
st.markdown('From the visualiztion, it shows mpg is higher when the car have less cylinders, displacement, horsepower and weight. \
            The tendency for MPG is droping with the year. So I recommend to buy newer cars if you want to save cost on gasoline.\
            The car in USA is more likely to have a small MGP than cars from Japan and has less horsepower in same MPG. Choose car from US if you presue large horsepower car with larger accelatration')

#######################################
###100,000 UK Used Car Dataset##########
########################################

st.header('100,000 UK Used Car Dataset')


# Read CSV files
df_ford = pd.read_csv('dataset/ford.csv')
df_ford["origin"] = "ford"

df_toyota = pd.read_csv('dataset/toyota.csv')
df_toyota["origin"] = "toyota"

# Concatenate DataFrames
df_UK_used = pd.concat([df_ford, df_toyota])

# Select numeric columns
numeric_columns = df_UK_used.select_dtypes(include=[np.number]).columns
df_UK_used_numeric = df_UK_used[numeric_columns]

# Check if there are any numeric columns
if df_UK_used_numeric.empty:
    raise ValueError("No numeric columns found in the DataFrame.")

# Select string/object columns
string_columns = df_UK_used.select_dtypes(include=[object]).columns
df_UK_used_string = df_UK_used[string_columns]

# Scale numeric data
scaler = StandardScaler()
df_UK_used_numeric_scaled = scaler.fit_transform(df_UK_used_numeric)
df_UK_used_numeric_scaled = pd.DataFrame(df_UK_used_numeric_scaled, columns=df_UK_used_numeric.columns)

# Check lengths
print(len(df_UK_used_numeric_scaled))
print(len(df_UK_used_string))

# Ensure both DataFrames have the same number of rows before concatenation
if len(df_UK_used_numeric_scaled) != len(df_UK_used_string):
    raise ValueError("Length of numeric scaled DataFrame does not match string DataFrame.")

# Concatenate scaled numeric and string DataFrames
df_UK_used = pd.concat([df_UK_used_numeric_scaled, df_UK_used_string.reset_index(drop=True)], axis=1)
#######################################
########### Overview     ###############
########################################
st.markdown('This dataset is a more recent dataset records the used car sale in UK for some common brand.\
            Since some features like price in this dataset have a large range, I will present the result after standard scaling.\
            Use the selectbox to explore the information recorded in dataset')
# Example options for the selectbox
options = ['mpg', 'year', 'price', 'mileage']

selected_option_x = st.selectbox('Please Choose first feature:', options)
selected_option_y = st.selectbox('Please Choose second feature:', options)

# Display the selected option
alt.renderers.enable('mimetype')
if selected_option_x != selected_option_y:
    chart = alt.Chart(df_UK_used).mark_point().encode(
        x=selected_option_x,
        y=selected_option_y,
        color='origin'
    ).properties(
        width=500,
        height=400
    ).interactive()
    st.altair_chart(chart)

else:
    chart_hist = alt.Chart(df_UK_used).mark_bar(opacity=0.7).encode(
        alt.X(selected_option_x +":Q",bin=alt.Bin(maxbins=20)),
        alt.Y('count()').stack(None),
        color=alt.Color('origin:N', scale=alt.Scale(domain=['ford', 'toyota'], range=['blue', 'green']))  # Custom colors
    ).properties(
        width=350,
        height=300
    ).interactive()

    chart_violin = alt.Chart(df_UK_used, width=100).transform_density(
        selected_option_x,
        as_=[selected_option_x, 'density'],
        groupby=['origin']
    ).mark_area(orient='horizontal').encode(
        alt.X('density:Q')
            .stack('center')
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0], grid=False, ticks=True),
        alt.Y(str(selected_option_x)+':Q'),
        alt.Color('origin:N'),
        alt.Column('origin:N')
            .spacing(0)
            .header(titleOrient='bottom', labelOrient='bottom', labelPadding=0)
    ).properties(
        width=100,
        height=300
    ).interactive()

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_hist)
    with col2:
        st.altair_chart(chart_violin)

plt.figure(figsize=(10, 6))

df_UK_used_toyota = df_UK_used[df_UK_used["origin"]=="toyota"]
df_UK_used_ford = df_UK_used[df_UK_used["origin"]=="ford"]

model_toyota = LinearRegression()
model_toyota.fit(np.array(df_UK_used_toyota['mileage']).reshape(-1, 1), np.array(df_UK_used_toyota['price']).reshape(-1, 1))  
X_toyota = np.linspace(df_UK_used_toyota['mileage'].min(), df_UK_used_toyota['mileage'].max(), 100).reshape(-1, 1)
y_toyota = model_toyota.predict(X_toyota)
plt.scatter(df_UK_used_toyota['mileage'], df_UK_used_toyota['price'], color='blue', label='toyota Data points')
plt.plot(X_toyota, y_toyota, color='red', label='toyota Regression line')

model_ford = LinearRegression()
model_ford.fit(np.array(df_UK_used_ford['mileage']).reshape(-1, 1), np.array(df_UK_used_ford['price']).reshape(-1, 1))  
X_ford = np.linspace(df_UK_used_ford['mileage'].min(), df_UK_used_ford['mileage'].max(), 100).reshape(-1, 1)
y_ford = model_ford.predict(X_ford)
plt.scatter(df_UK_used_ford['mileage'], df_UK_used_ford['price'], color='green', label='ford Data points')
plt.plot(X_ford, y_ford, color='black', label='ford Regression line')
plt.legend()
plt.xlabel('mileage')
plt.ylabel('price')
st.pyplot(plt)

#######################################
########### Narrative     ###############
########################################
st.subheader('Conclusion')
st.markdown('In this dataset, no data is missing. But given price and mileage have very different scale with mpg, I need first scale it.\
            I compare the residual value of used car from ford(USA) and toyota(Japan).\
            The conclusion still hold toyota, as a Japan brand, have more MPG. Also, it have more residual value than ford car in the same mileage\
            ')
