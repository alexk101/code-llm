import matplotlib.pyplot as plt
import seaborn as sns

from utils.data import get_language_info

# Get the programming language info from TIOBE
df = get_language_info()
print(df)

# Sort by popularity for better visualization
plot_df = df.sort("Ratings(%)", descending=True)

plt.figure(figsize=(6, 8))

# Seaborn barplot
sns.barplot(
    y=plot_df["Programming Language"],
    x=plot_df["Ratings(%)"],
    orient="h",
    palette="viridis",
)
plt.xlabel("Popularity (%)")
plt.ylabel("Programming Language")
plt.title("TIOBE Top Programming Languages")
plt.tight_layout()  # Leave space for the table
plt.savefig("test.png")
