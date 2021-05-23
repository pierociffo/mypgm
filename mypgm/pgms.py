from collections import deque
from graphviz import Digraph
from pgmpy.models.MarkovModel import MarkovModel
from mypgm.base import Factor, CPD, RandomVar
import numpy as np
import itertools
from itertools import chain
from collections import Iterable

class ProbabilityDistribution:

    def __init__(self, factors):
        check_factors = []
        for f in factors:
            if not isinstance(f, Factor):
                
                f = Factor([v for v in f])
                
            check_factors.append(f)
    
        self.factors = check_factors
                
    def reduce(self, evidence):
        factors = []
        for f in self.factors:
            ev = [(var, value) for (var, value) in evidence if var in f.scope]
            factors.append(f.reduce(ev))

        return ProbabilityDistribution(factors)

    def joint(self, normalize=True):
        f = Factor([], [1.0])
        for fi in self.factors:
            f = f * fi
            
        if normalize:
            f = f.normalize() 
        else:
            f = f

        return f

    def variables(self):
        return set([v for f in self.factors for v in f.scope])

    def __repr__(self):
        return ''.join([f.__repr__() for f in self.factors])
            
         
#pd seve per applicarve ve a bn     
class BayesianNetwork(ProbabilityDistribution):

    def __init__(self, factors):
        super(BayesianNetwork, self).__init__(factors)

    def graph(self):
        g = {v: set() for f in self.factors for v in f.scope}
        for f in self.factors:
            if len(f.scope) > 0:
                u = f.scope[0]
                for w in f.scope[1:]:
                    g[w].add(u)

        return g
    
    def viz(self):
        gg = Digraph('G')
        for f in self.factors:
            if len(f.scope) > 0:
                u = f.scope[0]
                for w in f.scope[1:]:
                    gg.edge(str(w), str(u))
                    
        return gg

    def check_DAG(self):
        return topological_sorting(self.graph())
    
    def nxg(self):
        return nx.DiGraph(self.graph())
        
    def add_table_to(self, factor, matrix):
        if not isinstance(factor, Factor):
            factor = Factor([v for v in factor])
        for f, k in zip(self.factors, range(len(self.factors))):
            if f.scope == factor.scope:
                if isinstance(matrix[0], Iterable):
                    f.values = np.array([item for sublist in matrix for item in sublist])
                    if f.values.size != np.prod(f.scope_dimensions()):
                        raise Exception('Incorrect table given scope')
                elif len(list(matrix)) != np.prod(f.scope_dimensions()):
                    raise Exception('Incorrect table given scope')
                else:
                    f.values = np.array(matrix)
                
                f = f.to_cpd()
                self.factors[k] = f

    
    def add_var_to(self, factor, var):
        if not isinstance(factor, Factor):
                factor = Factor([v for v in factor])
        for i, f in enumerate(self.factors):
            if f.scope == factor.scope:
                f_new = f.__add__(Factor(scope=[var]))
                
                self.factors[i]=f_new

        self.check_DAG()
                
    def add_factor(self, factor):
        if not isinstance(factor, Factor):
                factor = Factor([v for v in factor])
                
        self.factors.append(factor)
        self.check_DAG()


    def joint_probability(self, full_assg_map):
        p = 1
        
        full_assg_map_check = {}
        for k, v in full_assg_map.items():
            if not isinstance(k, RandomVar):
                k_ = RandomVar(k)
            else:
                k_ = k
                
            full_assg_map_check[k_] = v
            
                    
        for f in self.factors:  
            assg = [full_assg_map_check[v] for v in f.scope]
            p *= f.values[f.atoi(assg)]

        return p

    def dimension(self):
        d = 0

        for f in self.factors:
            if len(f.scope) > 0:
                d += (f.scope[0].k - 1) * np.prod([v.k for v in f.scope[1:]])

        return int(d)
    
    def get_markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        list(blanket_nodes): List of nodes contained in Markov Blanket

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = BayesianModel([('x', 'y'), ('z', 'y'), ('y', 'w'), ('y', 'v'), ('u', 'w'),
        ...                    ('s', 'v'), ('w', 't'), ('w', 'm'), ('v', 'n'), ('v', 'q')])
        >>> G.get_markov_blanket('y')
        ['s', 'u', 'w', 'v', 'z', 'x']
        """
        children = list(self.nxg().successors(node))
        parents = list(self.nxg().predecessors(node))
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(list(self.nxg().predecessors(child_node)))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.discard(node)
        return list(blanket_nodes)

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        EdgeView([('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')])
        """
        moral_graph = nx.Graph()
        moral_graph.add_nodes_from(self.nxg().nodes())
        moral_graph.add_edges_from(self.nxg().to_undirected().edges())

        for node in self.nxg().nodes():
            moral_graph.add_edges_from(
                itertools.combinations(list(self.nxg().predecessors(node)), 2)
            )

        return moral_graph
    
    def to_markov_model(self):
        """
        Converts bayesian model to markov model. The markov model created would
        be the moral graph of the bayesian model.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> mm = G.to_markov_model()
        >>> mm.nodes()
        NodeView(('diff', 'grade', 'intel', 'letter', 'SAT'))
        >>> mm.edges()
        EdgeView([('diff', 'grade'), ('diff', 'intel'), ('grade', 'letter'), ('grade', 'intel'), ('intel', 'SAT')])
        """
        moral_graph = self.moralize()
        #mm = MarkovModel(moral_graph.edges())
        #mm.add_nodes_from(moral_graph.nodes())
        #come passare dal grafico ai nuovi fattori?
        from mypgm.approximated import MarkovRF 
        mm = MarkovRF(self.factors)

        return mm

    

    #def add_values_names
    #check cpd (ma lasciare utility)
    #remove a factor or marginalize a variable
    #add dimensions
    # just one arrow

def topological_sorting(graph):
    """Code published by Alexey Kachayev"""
    GRAY, BLACK = 0, 1
    
    order, enter, state = deque(), set(graph), {}

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY:
                raise Exception("Cycle detected")
            if sk == BLACK:
                continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter:
        dfs(enter.pop())
    
    return list(order)

import networkx as nx

class InfluenceDiagram:
    
    def __init__(self, chance_factors, utility_factors, decision_factors):
        
        check_cf = []
        for f in chance_factors:

            if not isinstance(f, Factor):
                
                f = Factor([v for v in f])
            
                if not isinstance(f, CPD):
                
                    f = f.to_cpd()
            check_cf.append(f)
    
        self.chance_factors = check_cf
        
        check_uf = []
        for f in utility_factors:

            if not isinstance(f, Factor):
                f = Factor([v for v in f], [], mod = 'utility')
            check_uf.append(f)
            
            
        check_df = []
        for f in decision_factors:

            if not isinstance(f, Factor):
                f = Factor([v for v in f], [], mod = 'decision')
            check_df.append(f)
    
        self.utility_factors = check_uf
        self.decision_factors = check_df
        #self.check_DAG()
        self.typology = {'chance': self.chance_factors, 'utility': self.utility_factors, 'decision': self.decision_factors}
        
        self.d_set = set([f.scope[0] for f in self.decision_factors])
        self.v_set = set([v for f in self.chance_factors for v in f.scope if v not in self.d_set]) 
        self.u_set = set(['u'+str(i) for i in range(len(self.utility_factors))])
        
    def nxg(self):
        return nx.DiGraph(self.graph())    
        
    def add_table_to(self, factor, matrix, mod=None):
        if mod is None:
            self.mod = 'chance'
        else:
            self.mod = mod
            
        if not isinstance(factor, Factor):
            factor = Factor([v for v in factor], mod=mod)
        
        for f, k in zip(self.typology[factor.mod], range(len(self.typology[factor.mod]))):
            if f.scope == factor.scope:
                if isinstance(matrix[0], Iterable):
                    f.values = np.array([item for sublist in matrix for item in sublist])
                    if f.values.size != np.prod(f.scope_dimensions()):
                        raise Exception('Incorrect table given scope')
                elif len(list(matrix)) != np.prod(f.scope_dimensions()):
                    raise Exception('Incorrect table given scope')
                else:
                    f.values = np.array(matrix)
                
                if f.mod != 'utility':
                    f = f.to_cpd()
                
                self.typology[factor.mod][k] = f

    
    def add_var_to(self, factor, var, mod):
        if not isinstance(factor, Factor):
                factor = Factor([v for v in factor], mod=mod)
        for f in self.typology[mod]:
            if f.scope == factor.scope:
                self.typology[mod].remove(f)
                f = f.__add__(Factor(var))
                self.typology[mod].append(f)
                
        self.check_DAG()
                
    def add_factor(self, factor):
        if not isinstance(factor, Factor):
                factor = Factor([v for v in factor])
                
        self.factors.append(factor)
        self.check_DAG()
    
    def graph(self):
        # le ulility non tornano mai indietro (non sono random, le definisco attraverso altri fattori e basta)
        g = {v: set() for f in chain(self.chance_factors, self.decision_factors) for v in f.scope}
        for f in chain(self.chance_factors, self.decision_factors):
            if len(f.scope) > 0:
                u = f.scope[0]
                for w in f.scope[1:]:
                    g[w].add(u)
        i = 0       
        for u in self.utility_factors:
            u_name = 'u'+str(i)
            for h in u.scope:
                g[h].add(u_name) 
            i += 1

        return g
                                      

    def check_DAG(self):
        return topological_sorting(self.graph())
        
    def viz(self):
        gg = Digraph('G')
        
        for f in self.chance_factors:
            if len(f.scope) > 0:
                u = f.scope[0]
                gg.attr('node', shape='ellipse')
                gg.node(str(u))
                for w in f.scope[1:]:
                    for c in self.decision_factors:
                        if w == c.scope[0]:
                            gg.attr('node', shape='box')
                            break
                        else:
                            gg.attr('node', shape='ellipse')
                    gg.node(str(w))
                    gg.edge(str(w), str(u))
                    
        for d in self.decision_factors:
            gg.attr('node', shape='box')
            gg.node(str(d.scope[0]))
            for n in d.scope[1:]:
                gg.edge(str(n), str(d.scope[0]))
        
        pos = 0
        for u in self.utility_factors:
            gg.attr('node', shape='diamond')
            gg.node("Utility"+str(pos))

            for var in u.scope:
                gg.edge(str(var), "Utility"+str(pos))
            pos += 1
                        

                    
        return gg

