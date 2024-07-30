# # Set up and imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import seaborn as sns
import calendar
from matplotlib.ticker import ScalarFormatter
import dask
from dask import delayed, compute

df = pd.read_excel('dataset.xlsx')
df

df.info()

# # Data Exploration

# ## Top sellers

top_selling = df.groupby('Product Name').agg({'Quantity': 'sum'}).sort_values('Quantity',ascending=False)[:8]

top_selling

# ## Top revenue generators
top_revenue = pd.DataFrame(df.groupby('Product Name')['Sales'].sum().astype(float))
top_revenue.sort_values(by=['Sales'], inplace=True, ascending=False)
top_revenue.head(8).style.format({'Sales': '${:,.2f}'})

# ## Most profitable
most_profitable = pd.DataFrame(df.groupby(['Product Name'])['Profit'].sum().astype(float))
most_profitable.sort_values(by=['Profit'], inplace = True, ascending = False)
most_profitable.head(8).style.format({'Profit': '${:,.2f}'})


# ## Top selling categories
top_selling_categories = df.groupby(['Category'])['Sales'].sum().astype(float)
print(top_selling_categories.to_string(float_format='%.0f'))

# ## Most profitable categories
most_profitable_category = pd.DataFrame(df.groupby(['Category'])['Profit'].sum().astype(float))
most_profitable_category.sort_values(by=['Profit'], inplace = True, ascending = False)
most_profitable_category.style.format({'Profit': '${:,.2f}'})

# ## Top selling items in each category
df.groupby(["Category", "Sub-Category"], as_index=False)["Quantity"].count()

# ## Most profitable customer segments
most_profitable_customer_segments = pd.DataFrame(df.groupby(['Segment'])['Profit'].sum().astype(float))
most_profitable_customer_segments.sort_values(by=['Profit'], inplace = True, ascending = False)
most_profitable_customer_segments.style.format({'Profit': '${:,.2f}'})



# ## Top countries by revenue
top_sales_countries = pd.DataFrame(df.groupby('Country')['Sales'].sum().astype(float))
top_sales_countries.sort_values('Sales',inplace=True, ascending=False)
top_sales_countries.head(5).style.format({'Sales': '${:,.2f}'})


# ## Seasonality: "What are the peak sales months for different product categories?"
df['Year-Month'] = df['Order Date'].dt.to_period('M')

monthly_sales = df.groupby(['Category', 'Year-Month'])['Sales'].sum().reset_index()
peak_sales_months = monthly_sales.loc[monthly_sales.groupby('Category')['Sales'].idxmax()]
peak_sales_months


# ## Seasonality: "How do seasonal patterns vary across different markets?"
df['Month'] = df['Order Date'].dt.month
monthly_sales = df.groupby(['Market', 'Month'])['Sales'].sum().reset_index()
pivot_table = monthly_sales.pivot(index='Month', columns='Market', values='Sales')
plt.figure(figsize=(16, 10))
for region in pivot_table.columns:
    plt.plot(pivot_table.index, pivot_table[region], marker='o', label=region)

# Set the x-axis labels to month names
month_names = [calendar.month_name[i] for i in range(1, 13)]
plt.xticks(ticks=range(1, 13), labels=month_names)

plt.title('Average Monthly Sales by Market')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# ## Promotions Strategy: "How do promotional events impact sales in different seasons?"
# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Add Season column
df['Season'] = df['Month'].apply(get_season)

# Group by promotion status and season, then calculate the average sales
seasonal_discount_sales = df.groupby(['Season', 'Discount'])['Sales'].mean().reset_index()
seasonal_discount_sales['Discount (%)'] = seasonal_discount_sales['Discount'] * 100
pivot_table = seasonal_discount_sales.pivot(index='Discount (%)', columns='Season', values='Sales')
pivot_table.plot(figsize=(12, 6), marker='o')
plt.title('Impact of Discounts on Sales in Different Seasons')
plt.xlabel('Discount (%)')
plt.ylabel('Average Sales')
plt.legend(title='Season')
plt.grid(True)
plt.tight_layout()

plt.show()

# ## Strategy: What products are most likely to see increased demand during the upcoming holiday season?
# Extract month and year from Order Date
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# Define holiday season months - Nov, Dec - and filter data based on that
holiday_season_months = [11, 12]
holiday_season_data = df[df['Month'].isin(holiday_season_months)]

# Group by product and calculate total sales during the holiday season
product_sales = holiday_season_data.groupby('Product Name')['Sales'].sum().reset_index()

# Sort products by total sales
top_products = product_sales.sort_values(by='Sales', ascending=False).head(10)

# Build chart
plt.figure(figsize=(10, 6))
plt.barh(top_products['Product Name'], top_products['Sales'], color='red')
plt.xlabel('Sales ($)')
plt.ylabel('Product Name')
plt.title('Top Sellers during Holiday Season')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ## Analyze sales growth or decline by comparing sales data from different years.

annual_sales = df.groupby('Year')['Sales'].sum().reset_index()

# Calculate year over year growth
annual_sales['YoY Growth'] = annual_sales['Sales'].pct_change() * 100

# Plot the annual sales
years = annual_sales['Year'].tolist()
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(annual_sales['Year'], annual_sales['Sales'].astype(float), marker='o')
plt.title('Annual Sales')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.xticks(years)
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Plot year over year growth
plt.subplot(2, 1, 2)
plt.bar(annual_sales['Year'], annual_sales['YoY Growth'], color='purple')
plt.title('Year-over-Year Sales Growth')
plt.xlabel('Year')
plt.ylabel('Sales Growth Rate (%)')
plt.grid(True)
plt.xticks(years)

plt.tight_layout()
plt.show()

print(annual_sales.to_string(index=False, formatters={
    'Sales': '${:.2f}'.format,
    'YoY Growth': '{:.2f}%'.format
}))


# # Data Preprocessing & Analysis

# ## Category comparison - Technology vs Office Supplies

technology = df.loc[df['Category'] == 'Technology']
office = df.loc[df['Category'] == 'Office Supplies']
technology.shape, office.shape

columns = ['Row ID', 'Order ID', 'Product Name',  'Ship Mode', 'Ship Date', 'Customer ID', 'Customer Name', 'Region', 'Segment',  'City', 'State',  'Country', 'Product ID', 'Sub-Category', 'Category', 'Quantity', 'Discount', 'Profit']
technology.drop(columns, axis=1, inplace=True)
office.drop(columns, axis=1, inplace=True)

# Sort the dataframe based on Order Date
technology.sort_values('Order Date')
office.sort_values('Order Date')

# Group the dataframe based on Order Date and Sales, sum daily sales
technology = technology.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()
technology = technology.set_index('Order Date')
office = office.set_index('Order Date')

y_tech = technology['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()

technology = pd.DataFrame({'Order Date':y_tech.index, 'Sales':y_tech.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})

# Merge for comparison
merged = technology.merge(office, on='Order Date', how='inner')
merged.rename(columns={'Sales_x': 'Technology Sales', 'Sales_y': 'Office Supplies Sales'}, inplace=True)

merged.tail(10)

# ## Drop columns on main DF and prepare for ML

# Define columns to drop before training the model
columns = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'State', 'Region', 'Product Name',
           'Product ID', 'Shipping Cost',
           'Country', 'City', 'Category', 'Sub-Category',  'Quantity', 'Discount', 'Profit']

# Check to see if columns exist first (so this block can re-run without an error)
columns_to_drop = [col for col in columns if col in df.columns]

# Drop the columns
df.drop(columns_to_drop, axis=1, inplace=True)

# Show structure of remaining dataframe at this point
df.info()

# Check for null values
df.isna().sum()

# Sort the dataframe based on Order Date
df.sort_values('Order Date')

# Group the dataframe based on Order Date and Sales, sum daily sales
df = df.groupby('Order Date')['Sales'].sum().reset_index()

df

# ## Sales by year

df['Year'] = df['Order Date'].dt.year
sales_by_year = df.groupby('Year')['Sales'].sum()

# # Machine Learning Model Set Up and Fitting

# Indexing
df = df.set_index('Order Date')
df.index

# Resample sales, daily mean grouped by month start
y = df['Sales'].resample('MS').mean()

# Daily average sales starting from 2014 (when dataset begins)
y['2014':]

y.plot(figsize=(16, 8))

# ## Seasonal Decomposition

# **Trend**: long-term direction or movement in the data over time
# 
# **Seasonal**: The repeating patterns/cycles in data at regular intervals (ie. daily, monthly, yearly)
# 
# **Residual**: The remaining noise or irregular component after removing the trend and seasonal effects

# Seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
plt.show()


# ## Train model, compare actual/predicted results, forecast future sales

# Define ranges for p, d, q
p = d = q = range(0, 2)

# Generate all combinations of p, d, q
pdq = list(itertools.product(p, d, q))

# Generate all combinations of seasonal (p, d, q, s)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# Function to fit and evaluate a SARIMA model
@delayed
def evaluate_model(y, param, param_seasonal):
    try:
        model = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit()
        return (results.aic, param, param_seasonal, results)
    except Exception as e:
        return (float("inf"), param, param_seasonal, None)

# Check the length of the time series data
min_length = max(p) + max(d) + max(q) + 12

if len(y) < min_length:
    print("The length of the time series data is too short for the given parameter ranges")
else:
    # Create a list to hold all the delayed evaluation tasks
    tasks = [evaluate_model(y, param, param_seasonal) for param in pdq for param_seasonal in seasonal_pdq]

    # Compute the results in parallel
    results = compute(*tasks)

    # Find the best model based on AIC
    best_aic = float("inf")
    best_param = None
    best_param_seasonal = None
    best_model = None

    for result in results:
        aic, param, param_seasonal, model = result
        if aic < best_aic:
            best_aic = aic
            best_param = param
            best_param_seasonal = param_seasonal
            best_model = model

        # Output the parameters and their corresponding AIC value
        print(f'ARIMA parameters: {param}x{param_seasonal} ---> AIC: {aic}')

    # Print the best model parameters and AIC
    print(f'Best ARIMA parameters: {best_param}x{best_param_seasonal} ---> AIC: {best_aic}')

    # Show the best model results
    if best_model is not None:
        best_model_summary = best_model.summary()
        print(best_model_summary)

        # Get predictions from the best model
        pred = best_model.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

        # Calculate the confidence intervals for the predictions
        pred_ci = pred.conf_int()

        # Print confidence intervals
        print("Confidence intervals for predictions:")
        print(pred_ci)

        # Plot actual sales and predicted sales
        ax = y['2014':].plot(label='Observed')

        # Overlay predicted sales
        pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=.7, figsize=(14, 7))

        # Shade the area between the lower and upper confidence interval bounds
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='y', alpha=.3)

        ax.set_ylabel('Sales')
        ax.set_xlabel('Order Date')
        plt.legend()
        plt.show()

        # Generate forecasts for next 100 time steps
        pred_uc = best_model.get_forecast(steps=100)

        # Calculate confidence intervals
        pred_ci = pred_uc.conf_int()

        # Plot observed sales data
        ax = y.plot(label='Observed Sales', figsize=(14, 6))

        # Plot the forecasted mean sales values
        pred_uc.predicted_mean.plot(ax=ax, label='Forecasted Sales')

        # Shade the area between the lower and upper confidence interval bounds
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='y', alpha=0.3)

        ax.set_ylabel('Sales')
        ax.set_xlabel('Year')
        plt.legend()
        plt.show()
    else:
        print("No valid model was found.")

# ## Testing model accuracy and performance (MSE, RMSE)

# Predicted mean values of sales generated by SARIMA model
y_predicted = pred.predicted_mean

# Original sales data
y_true = y['2017-01-01':]

# Squared difference between each predicted value, and the corresponding true value
mse = ((y_predicted - y_true)**2).mean()
print('Mean square error (MSE):', round(mse, 3))

print('Root mean square error (RMSE):', round(np.sqrt(mse), 3))


