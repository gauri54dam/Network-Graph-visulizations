#!/usr/bin/env python
# coding: utf-8

# ## Network Visualization in Python
# 
# - Helper notebook for article of same name published on Medium.

# ### Import

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load and process data

# In[3]:


# load data
df = pd.read_csv("book1.csv")
# pick only important weights (hard threshold)
df = df.loc[df['weight']>10, :]
df


# In[4]:


# import
import networkx as nx
# load pandas df as networkx graph
G = nx.from_pandas_edgelist(df, 
                            source='Source', 
                            target='Target', 
                            edge_attr='weight')
print("No of unique characters:", len(G.nodes))
print("No of connections:", len(G.edges))


# ## Option 1: NetworkX

# In[5]:


# all graph options
graphs_viz_options = [nx.draw, nx.draw_networkx, nx.draw_circular, nx.draw_kamada_kawai, nx.draw_random, nx.draw_shell, nx.draw_spring]

# plot graph option
selected_graph_option = 0

# plot
plt.figure(figsize=(8,6), dpi=100) 
graphs_viz_options[selected_graph_option](G)


# ## Option 2: PyVis

# In[6]:


get_ipython().system('pip install pyvis')


# In[8]:


# import pyvis
from pyvis.network import Network
# create vis network
net = Network(notebook=True, width=1000, height=600)
# load the networkx graph
net.from_nx(G)
# show
net.show("example.html")


# In[ ]:


from inspect import getmembers
for x in getmembers(nx):
    if 'draw' in x[0]:
        print(x)


# ## Option 3: Visdcc in Dash
# 
# - See `dash_app.py` file for the demo.

# In[ ]:


get_ipython().system('pip install visdcc')


# In[ ]:


# imports
import dash
import visdcc
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# create app
app = dash.Dash()

# load data
df = pd.read_csv("book1.csv")
df = df.loc[df['weight']>10, :]
node_list = list(set(df['Source'].unique().tolist() + df['Target'].unique().tolist()))
nodes = [{'id': node_name, 'label': node_name, 'shape': 'dot', 'size': 7} for i, node_name in enumerate(node_list)]
# create edges from df
edges = []
for row in df.to_dict(orient='records'):
    source, target = row['Source'], row['Target']
    edges.append({
        'id': source + "__" + target,
        'from': source,
        'to': target,
        'width': 2,
    })

# define layout
app.layout = html.Div([
      visdcc.Network(id = 'net', 
                     data = {'nodes': nodes, 'edges': edges},
                     options = dict(height= '600px', width= '100%')),
      dcc.RadioItems(id = 'color',
                     options=[{'label': 'Red'  , 'value': '#ff0000'},
                              {'label': 'Green', 'value': '#00ff00'},
                              {'label': 'Blue' , 'value': '#0000ff'} ],
                     value='Red'  )             
])

# define callback
@app.callback(
    Output('net', 'options'),
    [Input('color', 'value')])
def myfun(x):
    return {'nodes':{'color': x}}

# define main calling
if __name__ == '__main__':
    app.run_server(debug=True)


# ### Network visualization using Jaal

# In[3]:


pip install jaal


# In[ ]:


# import
from jaal import Jaal
from jaal.datasets import load_got
# load the data
edge_df, node_df = load_got()

# init Jaal and run server
Jaal(edge_df, node_df).plot()


# ### PyDash

# In[ ]:


# from dash import Dash, html, dcc
# from dash.dependencies import Input, Output, State
# import visdcc

# app = Dash(__name__)
# app.layout = html.Div(...)

# @app.callback(...)
# def myfun(...):
#     ...
#     return ...

# if __name__ == '__main__':
#     app.run_server()


# In[ ]:


from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import visdcc

app = Dash(__name__)

app.layout = html.Div([
      visdcc.Network(id = 'net', 
                     options = dict(height= '600px', width= '100%')),
      dcc.Input(id = 'label',
                placeholder = 'Enter a label ...',
                type = 'text',
                value = ''  ),
      html.Br(),html.Br(),
      dcc.RadioItems(id = 'color',
                     options=[{'label': 'Red'  , 'value': '#ff0000'},
                              {'label': 'Green', 'value': '#00ff00'},
                              {'label': 'Blue' , 'value': '#0000ff'} ],
                     value='Red'  )             
])

@app.callback(
    Output('net', 'data'),
    [Input('label', 'value')])
def myfun(x):
    data ={'nodes':[{'id': 1, 'label':    x    , 'color':'#00ffff'},
                    {'id': 2, 'label': 'Node 2'},
                    {'id': 4, 'label': 'Node 4'},
                    {'id': 5, 'label': 'Node 5'},
                    {'id': 6, 'label': 'Node 6'}                    ],
           'edges':[{'id':'1-3', 'from': 1, 'to': 3},
                    {'id':'1-2', 'from': 1, 'to': 2},
                   {'id':'1-4', 'from': 1, 'to': 4}]
           }
    return data

@app.callback(
    Output('net', 'options'),
    [Input('color', 'value')])
def myfun(x):
    return {'nodes':{'color': x}}

if __name__ == '__main__':
    app.run_server()


# In[ ]:




