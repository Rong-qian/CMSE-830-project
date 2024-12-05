import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset



st.title('Advice for car purchase')
st.image("images.jpeg")

st.header('Introduction')

st.markdown('Buying a car is not an easy decision because there are so many cars in the market. '
            'They come from different car brands, countries, have different size, performance and look. '
            'Everyone have different needs and preference for car.'
            'In this app, you will find suggestions to help you choose a car. '
            'There are many car brands from different countries. Different brands have different technology routes and design principles. '
            'We will start with the MPG dataset involving cars from different countries. Let\'s first choose which country car you want to buy since considering their performance and fuel efficiency.')

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
st.markdown('First thing to check is what are in the MPG(miles per gallon) dataset.\
            In MPG dataset, it tells you the origin of the car and the name of the model.\
            Other features included are related to the mpg. Use the select box to view the relation between different features.\
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

st.subheader('Filling Missing value')


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

#st.subheader('Scale')

#st.markdown('Use the standard scale to handle the data.Then use the selectbox to see the difference before/after scaling')


numeric_columns = df_mpg.select_dtypes(include=[np.number]).columns
df_mpg_numeric = df_mpg[numeric_columns]
string_columns = df_mpg.select_dtypes(include=[object]).columns
df_mpg_string = df_mpg[string_columns]

scaler = StandardScaler() #Scale the data:standardize features by removing the mean and scaling to unit variance.

df_mpg_numeric_scaled = scaler.fit_transform(df_mpg_numeric) # Get Scaled  data
df_mpg_numeric_scaled = pd.DataFrame(df_mpg_numeric_scaled, columns=df_mpg_numeric.columns)

df_mpg_scaled = pd.concat([df_mpg_numeric_scaled, df_mpg_string], axis=1)


df_mpg_japan = df_mpg[df_mpg["origin"]=="japan"]
df_mpg_usa = df_mpg[df_mpg["origin"]=="usa"]

model_japan = LinearRegression()
model_japan.fit(np.array(df_mpg_japan['horsepower']).reshape(-1, 1), np.array(df_mpg_japan['mpg']).reshape(-1, 1))  
X_japan = np.linspace(df_mpg_japan['horsepower'].min(), df_mpg_japan['horsepower'].max(), 100).reshape(-1, 1)
y_japan = model_japan.predict(X_japan)

plt.clf()
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
st.subheader('Conclusion')
st.markdown('From the visualization, it shows mpg is higher when the car has fewer cylinders, displacement, horsepower and weight. \
            The tendency for MPG is dropping with the year. So I recommend to buy newer cars if you want to save cost on gasoline.\
            The car in USA is more likely to have a small MGP than cars from Japan and has less horsepower in the same MPG. Choose a car from US if you prefer large horsepower car with larger acceleration')

#######################################
###100,000 UK Used Car Dataset##########
########################################

st.header('Car brand comparison')

st.markdown('Now, we will use 100,000 UK Used Car Dataset to compare two popular brands, Toyota and Ford to help you choose between if you want to buy a used car')
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
st.markdown('I compare the residual value of used car from ford(USA) and toyota(Japan).\
            The conclusion still hold toyota, as a Japan brand, have more MPG. Also, it have more residual value than ford car in the same mileage\
            ')


#################################################
########### Car resale prediction ###############
#################################################

st.subheader('Resale value')
st.markdown('Now, you have a idea of which brand is better for your choice. \
            You may be interested in the resale value in case your demand for the car changes and need to buy a new one. \
            If you know what size, price, performance, etc. of your dream car, can we predict the resale value? \
            Next, I will build a model to predict the resale value by using the carsale dataset. \
            ')

##################################
# Load the dataset               #
##################################

st.markdown('First, Lets take a look at the carsales dataset. \
             We still use the select box to figure out the features, feature distribution and correlation.')

df_carsales_raw= pd.read_csv('./dataset/Car_sales.csv')
carsales_columns_to_use = ["Sales_in_thousands", "__year_resale_value","Price_in_thousands","Engine_size","Horsepower","Wheelbase","Width","Length","Curb_weight","Fuel_capacity","Fuel_efficiency","Power_perf_factor"]
df_carsales_select = df_carsales_raw[carsales_columns_to_use]

# Example options for the selectbox
options = carsales_columns_to_use

selected_option_x = st.selectbox('Choose first feature:', options)
selected_option_y = st.selectbox('Choose second feature:', options)

# Display the selected option
alt.renderers.enable('mimetype')
if selected_option_x != selected_option_y:
    chart = alt.Chart(df_carsales_select).mark_point().encode(
        x=selected_option_x,
        y=selected_option_y    
    ).properties(
        width=500,
        height=400
    ).interactive()
    st.altair_chart(chart)

else:
    chart_hist = alt.Chart(df_carsales_select).mark_bar(opacity=0.3).encode(
        alt.X(selected_option_x +":Q", bin=True),
        alt.Y('count()').stack(None)
    ).properties(
        width=350,
        height=300
    ).interactive()

    chart_violin = alt.Chart(df_carsales_select, width=100).transform_density(
        selected_option_x,
        as_=[selected_option_x, 'density']
    ).mark_area(orient='horizontal').encode(
        alt.X('density:Q')
            .stack('center')
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0], grid=False, ticks=True),
        alt.Y(str(selected_option_x)+':Q')
    ).properties(
        width=100,
        height=300
    ).interactive()

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_hist)
    with col2:
        st.altair_chart(chart_violin)


st.subheader('Prediction of carsales')

st.markdown('In carsale dataset, many resales value is missing. There are also few entries missing some of the car specifications \
             But since we want to predict resales value, we can fix it! \
             A simple prediction is based on MICE, we will use this linear model to fill/predict the missing value. See the comparison in test data ' )


########################################
#               Draw missing           #
########################################

# create a boolean mask: True for NaN, False for finite values
nan_mask = df_carsales_select.isna()

# convert boolean mask to integer (False becomes 0, True becomes 1)
nan_array = nan_mask.astype(int).to_numpy()

# size the plot
plt.figure(figsize=(12, 6))

# imshow with interpolation set to 'nearest' and aspect to 'auto'
im = plt.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')

plt.xlabel('Car Index')
plt.ylabel('Features')
plt.title('Visualizing Missing Values in a Dataset')

# y-axis tick labels to feature names
plt.yticks(range(len(df_carsales_select.columns)), df_carsales_select.columns)

# x-axis ticks
num_cars = nan_array.shape[0]
plt.xticks(np.linspace(0, num_cars-1, min(10, num_cars)).astype(int))

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

st.pyplot(plt)


########################################
#                 MICE                 #
########################################
data = df_carsales_select.copy()

train_features = ["Sales_in_thousands","Price_in_thousands","Engine_size","Horsepower","Wheelbase","Width","Length","Curb_weight","Fuel_capacity","Fuel_efficiency","Power_perf_factor"]

test_features = ['__year_resale_value']
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=['__year_resale_value'])

# Split features and target
X = data[train_features]
y = data[test_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform MICE imputation
mice_imputer = IterativeImputer(random_state=42, max_iter=20)
X_train_mice = pd.DataFrame(mice_imputer.fit_transform(X_train), 
                            columns=X_train.columns, index=X_train.index)
X_test_mice = pd.DataFrame(mice_imputer.transform(X_test), 
                           columns=X_test.columns, index=X_test.index)

# Train a linear regression model on the MICE imputed data
lr_mice = LinearRegression()
lr_mice.fit(X_train_mice, y_train)

# Make predictions and calculate MSE and R2
y_pred_mice = lr_mice.predict(X_test_mice)
mse_mice = mean_squared_error(y_test, y_pred_mice)
r2_mice = r2_score(y_test, y_pred_mice)

print(f"MICE Imputation Results:")
print(f"Mean Squared Error: {mse_mice:.4f}")
print(f"R2 Score: {r2_mice:.4f}")
# Make predictions and calculate MSE and R2
y_pred_mice = lr_mice.predict(X_test_mice)
mse_mice = mean_squared_error(y_test, y_pred_mice)
r2_mice = r2_score(y_test, y_pred_mice)

# Visualize the results
plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.scatter(y_test, y_pred_mice, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual resale value')
plt.ylabel('Predicted resale value')
plt.title('MICE Imputation: Predicted vs Actual')

plt.tight_layout()
st.pyplot(plt)


########################################
#              PCA                 #
########################################
st.subheader('SVD decomposition')

st.markdown('Now let\'s go back to predict the resale price for your dream car. \
                Before creating a model, we have too many features to consider. \
             Therefore, we can do PCA first to see what features are important \
             ')

# Load and prepare data
df_carsales_select_add = df_carsales_select.dropna().copy()

df_carsales_select_add[ 'residual'] = df_carsales_select_add[ '__year_resale_value']/df_carsales_select_add[ 'Price_in_thousands']
df_carsales_select_add_pca = df_carsales_select_add[["Price_in_thousands","Sales_in_thousands","Engine_size","Horsepower","Wheelbase","Width","Length","Curb_weight","Fuel_capacity","Fuel_efficiency","Power_perf_factor"]]

X = df_carsales_select_add_pca.copy()
X_scaled = StandardScaler().fit_transform(X)

# Perform PCA
U, s, Vt = np.linalg.svd(X_scaled)
V = Vt.T

# Create figure with both scree and biplot
fig = plt.figure(figsize=(15, 6))

# 1. Scree plot
plt.subplot(121)
var_exp = s**2 / np.sum(s**2) # Normalize to relative variance for each feature
cum_var_exp = np.cumsum(var_exp) # Cumsum of the relative variance

# Do actual plotting 
plt.plot(range(1, len(var_exp) + 1), var_exp, 'bo-', label='Individual') # Draw relative variance for each feature
plt.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, 'ro-', label='Cumulative')# Draw Cumulative variance for each feature
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.legend()
plt.grid(True)


# 2. Biplot
plt.subplot(122)
scores = X_scaled @ V
scale = 0.8  # Adjust this to change relative scaling of arrows

# Plot scores
plt.scatter(scores[:,0], scores[:,1], c='b', alpha=0.5, label='carsale ')

# Plot loadings
for i, feature in enumerate(df_carsales_select_add_pca.keys()):
    x = V[i,0] * s[0] * scale
    y = V[i,1] * s[1] * scale
    
    plt.arrow(0, 0, x, y, color='r', alpha=0.5, head_width=0.1)
    
    # Add labels with offset based on quadrant
    if x >= 0:
        ha = 'left'
    else:
        ha = 'right'
    if y >= 0:
        va = 'bottom'
    else:
        va = 'top'
        
    plt.text(x*1.1, y*1.1, feature, ha=ha, va=va)

plt.xlabel(f"PC1 ({var_exp[0]:.1%} variance)") # Variance of first feature
plt.ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
plt.title('PCA Biplot')
plt.grid(True)

# Add legend
plt.plot([0], [0], 'r-', label='Features')
plt.legend()

plt.tight_layout()
st.pyplot(plt)
# Print feature loadings for reference
st.write("\nFeature loadings (scaled by singular values):")
table_list = []
for name, v1, v2 in zip(df_carsales_select_add_pca.keys(), 
                       V[:,0] * s[0], 
                       V[:,1] * s[1]):
    table_list.append([name,v1,v2])

df_table = pd.DataFrame(table_list, columns=["name", "PC1","PC2"])
# Display the selected options in a table if any are selected
st.table(df_table)  # or use st.dataframe(df) for a more interactive table


########################################
#                 NN                 #
########################################

st.subheader('NN prediction')

st.markdown('By looking at the PCA output. Some features like price is quite important in the dataset. \
             Now instead of Linear Model we build in MICE to fill missing data, we will build Nerual network to predict the resale value. \
             You can choose the features you want to include in the training. Consider which feature is important based on the PCA result! \
             If you miss important features like the original price, you may get unreliable result.\
             ')

# Initialization
    

train_feature_options = ["Sales_in_thousands","Price_in_thousands","Engine_size","Horsepower","Wheelbase","Width","Length","Curb_weight","Fuel_capacity","Fuel_efficiency","Power_perf_factor"]

if 'renew' not in st.session_state:
        st.session_state['renew'] = 0
    
training = 0
input_options_old = []
selected_options= st.multiselect("Choose some options:", train_feature_options)
if selected_options:
    string_list = [str(option) for option in selected_options]
    training = 1
    selected_options = string_list 
    input_options = string_list
    if len(selected_options) !=0:
        st.session_state['renew'] = 1
    selected_options.append("__year_resale_value")
    input_options_old=input_options
else:
    st.write("No options selected.")
    

# Display the selected options
if st.session_state['renew']!=0:
    
    X_NN = df_carsales_select_add[selected_options]
    # Create a column of ones with the same number of rows

    scaler = StandardScaler()
    X_NN_scaled = scaler.fit_transform(X_NN)
    
    
    input_size = len(X_NN.columns) - 1  # Exclude the target column
    hidden_size = 32
    output_size = 1  # Predicting a single value
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 16
    
    
    
    # Scale back to original
    #X_original = scaler.inverse_transform(X_scaled)
    train_df, test_df = train_test_split(X_NN_scaled, test_size=0.2, shuffle=False)
    
    class TabularDataset(Dataset):
        def __init__(self, data):
            self.features = torch.tensor(data[:, :-1], dtype=torch.float32)
            self.target = torch.tensor(data[:, -1], dtype=torch.float32)
    
        def __len__(self):
            return len(self.features)
    
        def __getitem__(self, idx):
            x = self.features[idx]
            y = self.target[idx]
            return x, y
    
    # Prepare train and test datasets
    train_dataset = TabularDataset(train_df)
    test_dataset = TabularDataset(test_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define the neural network model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
    
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Initialize model, loss, and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()  # Using Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    # Training loop
    for epoch in range(num_epochs):
        # Lists to store actual and predicted values for plotting
        test_truth = []
        test_predictions = []
    
        model.train()
        for batch in train_dataloader:
            x, y = batch
            y = y.unsqueeze(1)  # Add a dimension for the target (since it's a single value)
            outputs = model(x)
            loss = criterion(outputs, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch in test_dataloader:
                x_test, y_test = batch
                y_test = y_test.unsqueeze(1)
                test_outputs = model(x_test)
                test_loss += criterion(test_outputs, y_test).item()
    
                # Append the true values and predicted values for plotting
                test_truth.extend(y_test.numpy())
                test_predictions.extend(test_outputs.numpy())
    
            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Test Loss after Epoch {epoch + 1}: {avg_test_loss:.4f}")
            st.session_state['model'] = "trained"
    

    test_truth = np.array(test_truth)
    test_predictions = np.array(test_predictions)

    mse = squared_sum = np.sum((test_truth - test_predictions) ** 2)/len(test_truth)
    st.session_state['model'] = "trained"
    # Plot the true vs predicted values
    plt.figure(figsize=(10, 6))
        
    plt.scatter(test_predictions,test_truth, label='True Values', color='blue', marker='o')
        #plt.plot(test_predictions, label='Predicted Values', color='red', marker='x', linestyle='solid', markersize=5)
    plt.title('Test Truth vs Predicted Values,MSE = {0}'.format(mse))
    plt.xlabel('Test predict')
    plt.ylabel('test truth')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Save model
    #torch.save(model.state_dict(), "mlp_model.pth")

########################################
#              Prediction              #
########################################

    
    
    st.write('By selecting different features, you can compare the different performances of the testing dataset.\
                 Now you can input features of your dream car, and get a prediction of the resale value. \
                 Hope it can help you make better decisions before purchasing a car. \
               ')
    
    # Title of the app
    st.title("Enter features value in the same sequence as you select your model")
    
    # Initialize an empty list to store the inputs
    input_values = []
    # Collect 3 float inputs using a for loop
    for i in range(len(input_options)-1):
        st.write("Input Feature" + input_options[i])

        value = st.number_input(f"Enter float value {i + 1}:", key=f"input_{i}", value=0.0, format="%.2f")
        input_values.append(value)

    my_input = torch.tensor(input_values)
    my_output = model(my_input)
    my_output.detach().numpy()[0]
    st.write( "You car resale value is (in k usd)",my_output.detach().numpy()[0])