import numpy as np
import random

class Node:
  def __init__(self, l, u):
    self.l = l
    self.u = u
    self.diam = np.max(u - l)
    self.midpoint = 0.5 * (l + u)
  def inside(self, l, u):
    if np.all(self.l >= l) and np.all(u >= self.u):
      return True
    else:
      return False


def branch(node):
  # Force to float to avoid truncation issues
  l = node.l.astype(float)
  u = node.u.astype(float)

  # Pick the dimension with largest length
  arg = np.argmax(u - l)
  mid = 0.5 * (l[arg] + u[arg])

  # Generate child boxes
  l1, u1 = l.copy(), u.copy()
  l2, u2 = l.copy(), u.copy()
  
  # Split the largest axis
  # along along midpoint of said axis
  u1[arg] = mid
  l2[arg] = mid
  node1 = Node(l1, u1)
  node2 = Node(l2, u2)
  return node1, node2


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if __name__ == "__main__":
  ndim = 2
  l0 = np.zeros(ndim)
  u0 = np.ones(ndim)
  root = Node(l0, u0)
  
  r1_l = np.zeros(ndim)
  r1_u = 0.33 * np.ones(ndim)

  r2_l = 0.66 * np.ones(ndim)
  r2_u = 1.0 * np.ones(ndim)

  node_list = [root]
  prunednode_list = []
  for i in range(1000):
    nnodes = len(node_list)
    idx = random.randint(0, nnodes-1)
    node = node_list.pop(idx)
    child1, child2 = branch(node)
    children = [child1, child2]
    if True:#i > 10:
      for j in range(2):  
        child = children.pop(0)
        if ((child.inside(r1_l, r1_u) or child.inside(r2_l, r2_u))):
          prunednode_list = prunednode_list + [child]
        else:
          children.append(child)
    node_list = node_list + children
  for node in node_list:
    X = np.linspace(node.l[0], node.u[0])
    Ylower = node.l[1] * np.ones(len(X))
    Yupper = node.u[1] * np.ones(len(X))
    plt.fill_between(X, Ylower, Yupper, color='lightblue', alpha=0.5)
  prunednode_midpoints = np.array([node.midpoint for node in prunednode_list])
  plt.scatter(prunednode_midpoints[:,0], prunednode_midpoints[:,1], color='red', marker='o', s=10, label='pruned nodes')
  plt.legend()
  #for node in prunednode_list:
    #X = np.linspace(node.l[0], node.u[0])
    #Ylower = node.l[1] * np.ones(len(X))
    #Yupper = node.u[1] * np.ones(len(X))
    #plt.fill_between(X, Ylower, Yupper, color='red', alpha=0.5)
  plt.show()

  midpoints = []
  for node in prunednode_list:
    midpoints.append(node.midpoint)
  midponts = np.array(midpoints)

  cluster_values = [k for k in range(2, 9)]
  silhouette_scores = []
  for k in cluster_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(midpoints)
    score = silhouette_score(midpoints, cluster_labels)
    silhouette_scores.append(score)
    print(f"The Silhouette score is: {score}, for {k} clusters")
  plt.plot(cluster_values, silhouette_scores)
  plt.xlabel('k (# of clusters)')
  plt.ylabel('Silhouette score')
  plt.show()
