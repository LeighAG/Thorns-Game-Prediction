import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('nwsl_data_cleaned.csv')

# Standardize column
df['Result'] = df['Result'].str.title()

# Streamlit app title
st.title("⚽ Portland Thorns FC: Exploratory Dashboard")

# Overview text explaining the data
st.write("""
### Overview:
This dashboard provides an exploratory analysis of Portland Thorns FC performance over recent seasons using data from Fbref. The data includes complete seasons from **2021 to 2024**, along with some incomplete data from the ongoing **2025** season. The goal is to build a predictive model for future match outcomes based on this performance data.

The dataset includes various match statistics such as goals scored (GF), expected goals (xG), shots on target (SoT%), and much more. Visualizations provided here will help identify patterns and trends in the team's performance across different metrics.
""")

# Sidebar season selector
seasons = sorted(df['Season'].unique())
selected_season = st.sidebar.selectbox("Select a Season", seasons)

# Filter data for the selected season
season_df = df[df['Season'] == selected_season]

# Summary stats
st.subheader(f"Season {selected_season} Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Goals", int(season_df['GF'].sum()))
col2.metric("Average xG", round(season_df['xG'].mean(), 2))
col3.metric("Win Rate", f"{(season_df['Result'] == 'Win').mean()*100:.1f}%")

# Text before visualizations
st.write("""
### Data Visualizations:
Below are some key visualizations that help explore the performance of Portland Thorns FC across different seasons. The data spans from **2021 to 2024**, with incomplete data for **2025**.
""")

# Line Graph: Win, Draw, Loss rates over seasons
st.subheader("📈 Win, Draw, and Loss Rates Over Seasons")
# Group by Season and calculate win/draw/loss rates as percentages
result_rates = df.groupby('Season')['Result'].value_counts(normalize=True).unstack(fill_value=0) * 100

# Extract each result rate, filling missing ones with 0s
win_rate = result_rates.get('W', pd.Series(0, index=result_rates.index))
draw_rate = result_rates.get('D', pd.Series(0, index=result_rates.index))
loss_rate = result_rates.get('L', pd.Series(0, index=result_rates.index))

# Reset index so we can plot by season properly
seasons = result_rates.index.tolist()

# Create line plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(seasons, win_rate, marker='o', label='Win %', color='green')
ax.plot(seasons, draw_rate, marker='o', label='Draw %', color='gray')
ax.plot(seasons, loss_rate, marker='o', label='Loss %', color='red')

ax.set_title('Win, Draw, and Loss Rates Over Seasons')
ax.set_xlabel('Season')
ax.set_ylabel('Percentage')
ax.set_xticks(seasons)
ax.set_xticklabels(seasons, rotation=45)
ax.legend()
ax.grid(True)

# Show the plot in Streamlit
st.pyplot(fig)

# Boxplot: Goals Scored by Venue
st.subheader("📊 Goals Scored by Venue (Home, Away, Neutral)")
# Define colors for all venue types
venue_palette = {'Home': 'blue', 'Away': 'red', 'Neutral': 'purple'}

# Create the Seaborn boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='Venue', y='GF', data=df, hue='Venue', palette=venue_palette, legend=False)

# Set plot title and labels
plt.title('Goals Scored by Venue')
plt.xlabel('Venue Type')
plt.ylabel('Goals Scored (GF)')

# Show the plot within Streamlit
st.pyplot(plt)

# Plot: Goals vs Expected Goals over seasons
st.subheader("📊 Goals Scored vs Expected Goals (All Seasons)")
fig1, ax1 = plt.subplots(figsize=(10, 5))
season_grouped = df.groupby('Season')[['GF', 'xG']].mean()
ax1.bar(season_grouped.index, season_grouped['GF'], width=0.4, label='Goals', color='blue', align='edge')
ax1.bar(season_grouped.index, season_grouped['xG'], width=-0.4, label='xG', color='orange', align='edge')
ax1.set_ylabel("Average Goals")
ax1.set_xlabel("Season")
ax1.legend()
st.pyplot(fig1)

# Plot: Boxplot of SoT% by Season
st.subheader("🎯 Distribution of Shots on Target % by Season")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Season', y='SoT%', hue='Season', palette='coolwarm', legend=False, ax=ax2)
ax2.set_ylabel("SoT%")
st.pyplot(fig2)

# Select the columns you want to display
columns_to_display = ['Date', 'Venue', 'Result', 'GF', 'GA', 'Opponent', 'Sh', 'SoT', 'xG', 'Poss']  # Customize this list as needed
df_filtered = df[columns_to_display]

# Data Preview
with st.expander("🔍 View Sample of Raw Data"):
    st.dataframe(df_filtered)
