import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pca_utils import plot_widget
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import plotly.offline as py
import matplotlib

matplotlib.use('TkAgg')
py.init_notebook_mode()
output_notebook()
X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])

plt.plot(X[:, 0], X[:, 1], 'ro')
plt.show()

pca_2 = PCA(n_components=2)
print(pca_2)
pca_2.fit(X)

print(pca_2.explained_variance_ratio_)

X_trans_2 = pca_2.transform(X)
print(X_trans_2)

pca_1 = PCA(n_components=1)
pca_1.fit(X)
print(pca_1.explained_variance_ratio_)
X_trans_1 = pca_1.transform(X)

X_reduced_2 = pca_2.inverse_transform(X_trans_2)
print(X_reduced_2)

plt.plot(X_reduced_2[:, 0], X_reduced_2[:, 1], 'ro')
plt.show()

X_reduced_1 = pca_1.inverse_transform(X_trans_1)
print(X_reduced_1)
plt.plot(X_reduced_1[:, 0], X_reduced_1[:, 1], 'ro')
plt.show()

#visualizing the PCA Algorithm

X = np.array([[-0.83934975, -0.21160323],
       [ 0.67508491,  0.25113527],
       [-0.05495253,  0.36339613],
       [-0.57524042,  0.24450324],
       [ 0.58468572,  0.95337657],
       [ 0.5663363 ,  0.07555096],
       [-0.50228538, -0.65749982],
       [-0.14075593,  0.02713815],
       [ 0.2587186 , -0.26890678],
       [ 0.02775847, -0.77709049]])

p = figure(title = '10-point scatterplot', x_axis_label = 'x-axis', y_axis_label = 'y-axis') ## Creates the figure object
p.scatter(X[:,0],X[:,1],marker = 'o', color = '#C00000', size = 5) ## Add the scatter plot

## Some visual adjustments
p.grid.visible = False
p.grid.visible = False
p.outline_line_color = None 
p.toolbar.logo = None
p.toolbar_location = None
p.xaxis.axis_line_color = "#f0f0f0"
p.xaxis.axis_line_width = 5
p.yaxis.axis_line_color = "#f0f0f0"
p.yaxis.axis_line_width = 5

## Shows the figure
show(p)
# plot_widget() issue, not workin

from pca_utils import random_point_circle, plot_3d_2d_graphs
X = random_point_circle(n = 150)
deb = plot_3d_2d_graphs(X)
deb.show()
deb.update_layout(yaxis2 = dict(title_text = 'test', visible=True))


#using PCA in data analysis
df = pd.read_csv(r"Unsupervised_Learning\data\toy_dataset.csv")
print(df.head())

#randomly pairing 100 sets of features (x, y)

def get_pairs(n = 100):
    from random import randint

    i = 0
    tuples = []
    while i < 100:
        x = df.columns[randint(0, 999)]
        y = df.columns[randint(0, 999)]
        while x == y or (x, y) in tuples or (y, x) in tuples:
            y = df.columns[randint(0, 999)]
        tuples.append((x, y))
        i += 1
    return tuples
pairs = get_pairs()
fig, axs = plt.subplots(10,10, figsize = (35,35))
i = 0
for rows in axs:
    for ax in rows:
        ax.scatter(df[pairs[i][0]],df[pairs[i][1]], color = "#C00000")
        ax.set_xlabel(pairs[i][0])
        ax.set_ylabel(pairs[i][1])
        i+=1
plt.show()

corr = df.corr()

mask = (abs(corr) > 0.5) & (abs(corr) != 1)
corr.where(mask).stack().sort_values()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(X_pca, columns=["principal_component_1", "principal_component_2"])
df_pca.head()
plt.scatter(df_pca['principal_component_1'],df_pca['principal_component_2'], color = "#C00000")
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA decomposition')
plt.show()
print(sum(pca.explained_variance_ratio_))
pca_3 = PCA(n_components = 3).fit(df)
X_t = pca_3.transform(df)
df_pca_3 = pd.DataFrame(X_t,columns = ['principal_component_1','principal_component_2','principal_component_3'])
import plotly.express as px
fig = px.scatter_3d(df_pca_3, x = 'principal_component_1', y = 'principal_component_2', z = 'principal_component_3').update_traces(marker = dict(color = "#C00000"))
fig.show()
