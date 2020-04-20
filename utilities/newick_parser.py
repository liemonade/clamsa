import newick as nwk
import numpy as np
from . import msa_converter as mc




def nwk_read(nwk_filename):
    """
        Reads a Newick file and processes its first tree.

        Args:
            nwk_filename (str): Path to the wanted Newick file

        Returns:
            List[Newick.Node]: A list of the nodes of the tree. These are in a specfic order dictated by their appearence in the file.
                                Specifically if the tree has `k` leaves then the first `k` entries of this list are the leaves.
            List[Tuple[int]]:  A list of tuples of the form `(v,w)` representing an edge from the node of index `v` to the node of index `w`
                               in the tree. These are in a particular order dictated by the order of nodes.
            numpy.array:       A matrix `A` with `m` rows and `t+1` columns, where `m` is the number of edges in the tree.
                               Given a set of `t` real numbers `x` meeting some constraints one can calculate 
                               `lengths = np.dot(A, np.c_[x,1])`
                               in order to compute an array of length `m` representing a valid edge length configuration for the input tree.

    """

    root = nwk.read(nwk_filename)[0]

    # first, add all nodes and edges into arrays
    nodes = []
    edges = []

    def parse_subtree(node):
        nonlocal nodes
        nonlocal edges

        if not node in nodes:
            nodes = nodes + [node]
            for child in node.descendants:
                parse_subtree(child)
                edges = edges + [(nodes.index(node),nodes.index(child))]

    parse_subtree(root)
    names = [v.name for v in nodes]
    leave_names = mc.leave_order(nwk_filename)

    
    
    # get a buttom up order of the nodes
    V = [nodes[names.index(l)] for l in leave_names]
    n = len(nodes)
    k = len(leave_names)
    
    for it in range(n):
        v = V[it]
        w = v.ancestor
        
        if w != None:
            
            # add parent to nodes if note present
            if w not in V:
                V.append(w)
            else:
                
                # start repairing indicies by a bubble sort like algorithm
                a = w
                while a != None and a.ancestor in V and V.index(a) > V.index(a.ancestor):
                    i = V.index(a)
                    j = V.index(a.ancestor)
                    V[i] = a.ancestor
                    V[j] = a
                    a = a.ancestor
            
                # check whether the "new" index of w needs to be switched
                if V.index(v) > V.index(w):
                    i = V.index(v)
                    j = V.index(w)
                    V[i] = w
                    V[j] = v
            
    
    E = sorted([(V.index(nodes[v]), V.index(nodes[w])) for (v,w) in edges])
    
    # calculate the needed variables and dependencies for edge lengths
    T = np.zeros((len(E), len(E)))
    root_var = len(E)-1
    vars_subtree = [[] for e in E]
    root_id = len(V) - 1
    t = 0
    
    for i in range(k):
        
        v = V[i]
        
        while V.index(v) != root_id:
            
            w = v.ancestor 
            i_v = V.index(v)
            i_w = V.index(w)
            e = E.index((i_w,i_v))
            
            if w == root:
                T[e,root_var] = 1
                T[e,vars_subtree[i_v]] = -1
            
            else:
                
                if vars_subtree[i_w] == []:
                    vars_subtree[i_w] = [t] + vars_subtree[i_v]
                    T[e,t] = 1
                    t = t+1
                else: 
                    T[e,vars_subtree[i_w]] = 1
                    T[e,vars_subtree[i_v]] = -1
                    break
            
            v = w
            
    T = np.c_[T[:,:t], T[:,-1]]
    
            
    return V, E, T


