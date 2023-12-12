import matplotlib.pyplot as plt

# Sample feature importances (replace with your actual data)
xgboost_importances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.6, 0.2, 0.4]
random_forest_importances = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.25, 0.35, 0.45, 0.55, 0.65, 0.45, 0.55, 0.65, 0.35, 0.45, 0.55, 0.65, 0.25, 0.45]
features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E", "Feature F", "Feature G", "Feature H", "Feature I", "Feature J", "Feature K", "Feature L", "Feature M", "Feature N", "Feature O", "Feature P", "Feature Q", "Feature R", "Feature S", "Feature T"]


# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(xgboost_importances[:20], random_forest_importances[:20])
plt.xlabel("XGBoost Importance")
plt.ylabel("Random Forest Importance")
for i, txt in enumerate(features[:20]):
    plt.annotate(txt, (xgboost_importances[i], random_forest_importances[i]))
plt.title("Scatter Plot of Feature Importances")
plt.show()

# Heatmap
import seaborn as sns
import pandas as pd

data = pd.DataFrame({"Feature": features[:20], "XGBoost": xgboost_importances[:20], "Random Forest": random_forest_importances[:20]})
data = data.set_index("Feature")

plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True, cmap="YlGnBu")
plt.title("Feature Importance Heatmap")
plt.show()

# Radar Chart
from math import pi

categories = features[:20]
N = len(categories)

xgboost_values = xgboost_importances[:20]
random_forest_values = random_forest_importances[:20]

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='grey', size=8)

ax.set_rlabel_position(0)
plt.yticks([0.1, 0.3, 0.5], ["0.1", "0.3", "0.5"], color="grey", size=7)
plt.ylim(0, 0.7)

# Make sure both lists have the same number of elements
xgboost_values += [xgboost_values[0]]
random_forest_values += [random_forest_values[0]]

ax.plot(angles, xgboost_values, linewidth=1, linestyle='solid', label='XGBoost')
ax.fill(angles, xgboost_values, 'b', alpha=0.1)

ax.plot(angles, random_forest_values, linewidth=1, linestyle='solid', label='Random Forest')
ax.fill(angles, random_forest_values, 'r', alpha=0.1)

plt.title("Radar Chart of Feature Importances")
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()



