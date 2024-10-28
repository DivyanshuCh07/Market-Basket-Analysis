# (1) Command :
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

data = pd.read_csv("market_basket_dataset.csv")
print(data.head())

# (2) Command :
print(data.isnull().sum())

# (3) Command :
print(data.describe())

# (4) Command:
fig = px.histogram(data, x='Itemname', 
                   title='Item Distribution')
fig.show()

# (5) Command:
	# Calculate item popularity
item_popularity = data.groupby('Itemname')['Quantity'].sum().sort_values(ascending=False)

top_n = 10
fig = go.Figure()
fig.add_trace(go.Bar(x=item_popularity.index[:top_n], y=item_popularity.values[:top_n],
	                     text=item_popularity.values[:top_n], textposition='auto',
	                     marker=dict(color='skyblue')))
fig.update_layout(title=f'Top {top_n} Most Popular Items',
	                  xaxis_title='Item Name', yaxis_title='Total Quantity Sold')
fig.show()

# (6) Command:
	# Calculate average quantity and spending per customer
customer_behavior = data.groupby('CustomerID').agg({'Quantity': 'mean', 'Price': 'sum'}).reset_index()

	# Create a DataFrame to display the values
table_data = pd.DataFrame({
	    'CustomerID': customer_behavior['CustomerID'],
	    'Average Quantity': customer_behavior['Quantity'],
	    'Total Spending': customer_behavior['Price']
	})

	# Create a subplot with a scatter plot and a table
fig = go.Figure()
	
	# Add a scatter plot
fig.add_trace(go.Scatter(x=customer_behavior['Quantity'], y=customer_behavior['Price'],
	                         mode='markers', text=customer_behavior['CustomerID'],
	                         marker=dict(size=10, color='coral')))

	# Add a table
fig.add_trace(go.Table(
	    header=dict(values=['CustomerID', 'Average Quantity', 'Total Spending']),
	    cells=dict(values=[table_data['CustomerID'], table_data['Average Quantity'], table_data['Total Spending']]),
	))

	# Update layout
fig.update_layout(title='Customer Behavior',
	                  xaxis_title='Average Quantity', yaxis_title='Total Spending')

	# Show the plot
fig.show()

# (7) Command:
from mlxtend.frequent_patterns import apriori, association_rules

	# Group items by BillNo and create a list of items for each bill
basket = data.groupby('BillNo')['Itemname'].apply(list).reset_index()

	# Encode items as binary variables using one-hot encoding
basket_encoded = basket['Itemname'].str.join('|').str.get_dummies('|')

	# Find frequent itemsets using Apriori algorithm with lower support
frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)

	# Generate association rules with lower lift threshold
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)

	# Display association rules
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

