import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Strip extra spaces in column names
df.columns = df.columns.str.strip()

# Drop rows where all values are NaN (you had 28 empty rows)
df.dropna(how='all', inplace=True)

# Rename columns for simplicity
df.rename(columns={
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

# Convert date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Preview
print(df.head())
print(df.info())

# Plot: Unemployment Rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Unemployment_Rate', ci=None)
plt.title("Unemployment Rate Over Time (India)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap by State and Date
pivot = df.pivot_table(values='Unemployment_Rate', index='State', columns='Date')
plt.figure(figsize=(14, 10))
sns.heatmap(pivot, cmap="coolwarm", linewidths=0.2)
plt.title("Unemployment Rate Heatmap by State and Date")
plt.tight_layout()
plt.show()

# Top 10 states with highest avg unemployment
top_states = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_states.values, y=top_states.index, palette="magma")
plt.title("Top 10 States by Average Unemployment Rate")
plt.xlabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()
