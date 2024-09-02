import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the DataFrame based on the updated image data
data = {
    'Tenors': ['1Yr', '2Yr', '3Yr', '4Yr', '5Yr', '8Yr', '9Yr', '10Yr', '15Yr', '20Yr', '30Yr'],
    'Coupon': [5.0704, 4.7023, 4.4515, 4.2825, 4.1697, 4.0333, 4.0274, 4.0327, 4.0908, 4.1097, 4.0639],
    '3Mo': [4.8433, 4.5258, 4.3180, 4.1765, 4.0837, 3.9850, 3.9865, 3.9979, 4.0853, 4.0874, 4.0433],
    '6Mo': [4.42, 4.3520, 4.1845, 4.0708, 3.9980, 3.9368, 3.9458, 3.9632, 4.0732, 4.0651, 4.0229],
    '1Yr': [4.09, 4.1238, 3.9990, 3.9224, 3.8779, 3.8744, 3.8951, 3.9219, 4.0095, 4.0372, 3.9949],
    '2Yr': [3.9227, 3.8305, 3.7808, 3.7577, 3.7583, 3.8333, 3.8696, 3.8985, 3.9917, 4.0163, 3.9657],
    '3Yr': [3.7341, 3.7056, 3.6987, 3.7138, 3.7443, 3.8620, 3.8957, 3.9269, 4.0080, 4.0230, 3.9575],
    '4Yr': [3.6760, 3.6797, 3.7058, 3.7470, 3.7887, 3.9187, 3.9519, 3.9786, 4.0269, 4.0434, 3.9595],
    '5Yr': [3.6836, 3.7215, 3.7723, 3.8193, 3.8695, 3.9923, 4.0186, 4.0394, 4.0812, 4.0689, 3.9649],
    '10Yr': [4.2125, 4.2126, 4.2304, 4.2394, 4.2447, 4.2273, 4.2241, 4.2216, 4.1652, 4.0908, 3.9353],
    '15Yr': [4.1935, 4.1934, 4.1934, 4.1935, 4.1934, 4.1326, 4.1213, 4.1124, 4.1124, 3.9004, 3.7287],
    '30Yr': [3.4032, 3.4031, 3.4031, 3.4031, 3.4031, 3.4031, 3.4031, 3.4031, 3.2544, 3.1798, 3.1059]
}

forwards = pd.DataFrame(data)

# Define all periods including the spot rates
all_periods = ['Coupon', '3Mo', '6Mo', '1Yr', '2Yr', '3Yr', '4Yr', '5Yr', '10Yr', '15Yr', '30Yr']

# Initialize a DataFrame to store roll calculations
rolls_data = pd.DataFrame(index=forwards['Tenors'], columns=all_periods)

# Calculate the roll for each combination of tenor and forward period
for i, tenor in enumerate(forwards['Tenors']):
    for j in range(1, len(all_periods)):
        rolls_data.at[tenor, all_periods[j]] = forwards.at[i, all_periods[j]] - forwards.at[i, all_periods[j-1]]

# Drop the initial column as it is not a roll value
rolls_data = rolls_data.drop(columns='Coupon')

# Reorder the columns and rows to match the desired order
rolls_data = rolls_data[all_periods[1:]]
rolls_data = rolls_data.reindex(['1Yr', '2Yr', '3Yr', '4Yr', '5Yr', '8Yr', '9Yr', '10Yr', '15Yr', '20Yr', '30Yr'])

# Set up the matplotlib figure
plt.figure(figsize=(16, 10))

# Generate a heatmap
sns.heatmap(rolls_data.astype(float), annot=True, fmt=".4f", cmap='coolwarm', center=0, linewidths=.5, cbar_kws={'label': 'Roll Value'})

# Add title and labels
plt.title('Heat Map of Roll Values Across Tenors and Forward Periods')
plt.xlabel('Forward Period')
plt.ylabel('Tenor')

# Show the plot
plt.show()
