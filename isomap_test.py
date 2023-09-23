from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
import pandas as pd
import plotly.express as px

# X, _ = load_digits(return_X_y=True)
# X.shape

# embedding = Isomap(n_components=2)
# X_transformed = embedding.fit_transform(X)
# X_transformed.shape

df1 = pd.read_csv('isomap_tp_1.csv')
fig = px.scatter(x=df1['0'], y=df1['1'])
fig.show()