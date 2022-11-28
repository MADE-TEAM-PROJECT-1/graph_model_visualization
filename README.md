# graph_model_visualization
# Description
Graph model is visualized using pyvis library.

The graph consist from:
- entities (vertices): articles and authors.
- relations (edges):
    - REFERENCE: edge between $article_1$ and $article_2$ if $article_1$ has reference to $article_2$;
    - AUTHOR: edge between $author$ and $article$ if $author$ is and author of $article$;
    - COAUTHOR: edge between $author_1$ and $author_2$ if they are coauthors (exist an article where they are authors).

Usage:

   ```
    from graph_model import graph_model
    g = graph_model(path_to_articles, size_cut)
   ```
where:
- path_to_articles : str Path to the csv data file with articles information
- size_cut : int Number of articles which should be used for graph model
 ```
    from graph_model import graph_model
    g = graph_model(path_to_articles, size_cut)
    g.build_graph(start_ver='', edge_type='COAUTHOR', max_dfs_depth=10, use_weights=True)
  ```
where:
- start_ver : str ID of the vertex, from which we start to make graph. If missing we are using all of the vertices
- edge_type : {'COAUTHOR', 'REFERENCE'} one of two types of edges we are using. 'REFERENCE' for graph of citation and  'REFERENCE' for coatuhorship graph
- max_dfs_depth : int the maximum depth of the graph starting from start_ver is visualized
- use_weights : bool should we visualize the weight of author as the number of articles they have written.
 ```
    from graph_model import graph_model
    g = graph_model(path_to_articles, size_cut)
    g.save_graph(graph_path: str)
  ```
where:
- graph_path : str the path for html with graph for saving
