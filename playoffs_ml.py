import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

folder_path = 'your_folder_path/playoffs'

all_data = []
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        year = int(file[:4])
        df = pd.read_csv(os.path.join(folder_path, file))
        df["Year"] = year
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

df_all.columns = df_all.columns.str.strip()
df_all = df_all[df_all['Player'] != 'Player']
df_all['Player'] = df_all['Player'].str.replace('*', '', regex=False).str.strip()
df_all['MP'] = pd.to_numeric(df_all['MP'], errors='coerce')
df_all = df_all[df_all['MP'] >= 100]

metrics = ['PER', 'WS', 'BPM', 'TS%']
df_all = df_all[['Player', 'Year'] + metrics]
df_all[metrics] = df_all[metrics].apply(pd.to_numeric, errors='coerce')
df_all.dropna(inplace=True)

df_player_avg = df_all.groupby('Player')[metrics].mean().reset_index()

scaler = MinMaxScaler()
df_player_avg[metrics] = scaler.fit_transform(df_player_avg[metrics])
df_player_avg['Score'] = df_player_avg[metrics].mean(axis=1)

top_10_players = df_player_avg.sort_values('Score', ascending=False).head(10).reset_index(drop=True)
print("Top 10 Players in the 2010–2020 Playoffs (Based on Overall Average Score):")
print(top_10_players[['Player', 'Score']])

kmeans = KMeans(n_clusters=4, random_state=42)
df_player_avg['Cluster'] = kmeans.fit_predict(df_player_avg[metrics])

cluster_order = df_player_avg.groupby('Cluster')['Score'].mean().sort_values(ascending=False).index
label_map = {
    cluster_order[0]: 'Elite',
    cluster_order[1]: 'Good',
    cluster_order[2]: 'Average',
    cluster_order[3]: 'Below Average'
}
df_player_avg['Label'] = df_player_avg['Cluster'].map(label_map)

plt.figure(figsize=(12, 8))

for label in df_player_avg['Label'].unique():
    subset = df_player_avg[df_player_avg['Label'] == label]
    plt.scatter(subset['PER'], subset['WS'], label=label, alpha=0.7)

elite_players = df_player_avg[df_player_avg['Label'] == 'Elite']
for i, row in elite_players.iterrows():
    plt.text(row['PER'], row['WS'], row['Player'], fontsize=8, alpha=0.9)

plt.xlabel("PER")
plt.ylabel("WS")
plt.title("Player Performance Groups with K-Means Clustering (Elite Players Labeled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()