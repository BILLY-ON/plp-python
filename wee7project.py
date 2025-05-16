# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    data['species'] = data['species'].map(dict(zip(range(3), iris.target_names)))

    print("Dataset loaded successfully!\n")
    print("First 5 rows of the dataset:")
    print(data.head())

except FileNotFoundError:
    print("Error: The dataset file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Check data types and missing values
print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

# No missing values in this dataset, so no cleaning needed.

# Task 2: Basic Data Analysis
print("\nDescriptive statistics:")
print(data.describe())

# Group by species and calculate mean of features
grouped = data.groupby('species').mean()
print("\nMean values by species:")
print(grouped)

# Task 3: Data Visualization

# 1. Line Chart - simulate time-series by plotting mean petal length per "index" per species
plt.figure(figsize=(8, 5))
for species in data['species'].unique():
    species_data = data[data['species'] == species]
    plt.plot(species_data.index, species_data['petal length (cm)'], label=species)
plt.title('Petal Length Over Index (Simulated Time)')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette='Set2')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.tight_layout()
plt.show()

# 3. Histogram - distribution of sepal length
plt.figure(figsize=(6, 4))
sns.histplot(data['sepal length (cm)'], kde=True, color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.tight_layout()
plt.show()

# 4. Scatter Plot - sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=data, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs. Petal Length')
plt.tight_layout()
plt.show()

# Findings:
print("\nObservations:")
print("- Iris-virginica tends to have the longest petals and sepals.")
print("- The species are fairly distinguishable based on petal length.")
print("- Sepal length and petal length are positively correlated.")

