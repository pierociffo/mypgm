from functools import reduce
from operator import mod
from mypgm.base import Factor, CPD, RandomVar
import numpy as np
from mypgm.pgms import ProbabilityDistribution, InfluenceDiagram
from itertools import chain

class VariableElimination:

    def __init__(self, gd):
        self.gd = gd

    def marginal(self, hypothesis, evidence=[], normalize=True):
        gd = self.gd.reduce(evidence)
        elim_variables = set(
            [v for f in gd.factors for v in f.scope if v not in hypothesis])

        gm = MinFill(gd.factors)
        elim_variables = gm.ordering(elim_variables)

        factors = gd.factors

        for v in elim_variables:
            factor = reduce(lambda x, y: x * y,
                            [f for f in factors if v in f.scope],
                            Factor([], np.array([1.0])))
            factor = factor.marginalize(v)
            factors = [f for f in factors if v not in f.scope] + [factor]

        return ProbabilityDistribution(factors).joint(normalize)

    def map_query(self, evidence=[]):
        gd = self.gd.reduce(evidence)

        elim_variables = set([v for f in gd.factors for v in f.scope])

        gm = MinFill(gd.factors)
        elim_variables = gm.ordering(elim_variables)

        factors = gd.factors

        phis = []
        for v in elim_variables:
            factor = reduce(lambda x, y: x * y,
                            [f for f in factors if v in f.scope],
                            Factor([], np.array([1.0])))

            phis.append(factor)

            factor = factor.maximize(v)
            factors = [f for f in factors if v not in f.scope] + [factor]

        assgmap = dict()
        for (i, f) in reversed(list(enumerate(phis))):
            assg = [0] * len(f.scope)

            for j, v in enumerate(f.scope):
                if v == elim_variables[i]:
                    assg[j] = -1
                else:
                    assg[j] = assgmap[v]

            assgmap[elim_variables[i]] = f.argmax(assg)

        return [(v, assgmap[v]) for v in elim_variables]

#comment this heuristic
class MinFill():

    def __init__(self, factors):
        self.m = {}
        for f in factors:
            for v in f.scope:
                if v in self.m:
                    self.m[v].update(f.scope)
                else:
                    self.m[v] = set(f.scope)

        for v in self.m.keys():
            self.m[v].remove(v)

    def ordering(self, elim_variables):
        ordering = []

        elim = list(elim_variables)
        m = dict(self.m)

        while elim:
            bestv, bestunc = elim[0], self.unconnected(m, m[elim[0]])

            for v in elim[1:]:
                unconnected = self.unconnected(m, m[v])

                if len(unconnected) < len(bestunc):
                    bestv = v
                    bestunc = unconnected

            ordering.append(bestv)
            elim.remove(bestv)

            del m[bestv]
            for v, n in m.items():
                if bestv in n:
                    n.remove(bestv)

            for u, v in bestunc:
                m[u].add(v)
                m[v].add(u)

        return ordering

    def unconnected(self, m, variables):
        unc = set()
        for u in variables:
            for v in variables:
                if u != v and v not in m[u]:
                    unc.add((min(u, v), max(u, v)))

        return unc

    def __repr__(self):
        return self.m.__repr__()


class ExpectedUtility:

    def __init__(self, influence_diagram):
        self.id = influence_diagram
    
    def expected_utility(self, decision_rule):
        eu = 0.0

        cf = self.id.chance_factors
        df = decision_rule

        for uf in self.id.utility_factors:
            gd = ProbabilityDistribution(cf + df + [uf])
            ve = VariableElimination(gd)

            eu += ve.marginal([], normalize=False).values[0]

        return eu

    # si potrebbe trovare un algoritmo che risolve tutto passando i diversi decision scope quando ce n'è più di uno
    def optimal_decision_rule(self, scope, others=None):
        cf = self.id.chance_factors
        uf = Factor([], [0.0], mod='utility')
        for f in self.id.utility_factors:
            uf += f
        
        if others is not None:
            if others:
                gd = ProbabilityDistribution(cf + others + [uf])
            else:
                gd = ProbabilityDistribution(cf + [uf])
        else:
            gd = ProbabilityDistribution(cf + [uf])

        
        ve = VariableElimination(gd)

        mu = ve.marginal(scope, normalize=False)
        assg_map = [scope.index(v) for v in mu.scope]
        ind = mu.scope.index(scope[0])

        rule = Factor(scope, np.zeros(np.prod([v.k for v in scope])), mod='decision')
        n = int(np.prod([v.k for v in scope[1:]]))

        for i in range(n):
            assg = rule.itoa(i)

            assg_mu = np.array(assg)[assg_map]
            assg_mu[ind] = -1

            assg_max = [mu.argmax(assg_mu)] + list(assg[1:])

            rule.values[rule.atoi(assg_max)] = 1.0

        return CPD(rule.scope, rule.values, mod='decision')






import networkx as nx
from collections import Counter

def find_decision_path(i_d):
    d_set = i_d.d_set
    g = i_d.nxg()

    roots = []
    leaves = []
    for node in g.nodes:
        if g.in_degree(node) == 0 : # it's a root
            roots.append(node)
        elif g.out_degree(node) == 0 : # it's a leaf
            leaves.append(node)

    paths = []
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(g, root, leaf):
                paths.append(path)
              
    good_path = False        
    for p in paths:
        decision_order = sorted(set(p) & d_set, key = p.index) 
        #decision_order = list(set(p) & d_set) 
        if Counter(decision_order) == Counter(list(d_set)):
            good_path = True
            
            break
                
    return good_path, decision_order



def random_decision(decision_factor):
    f = decision_factor
    div = f.values.size // f.scope[0].k
    data = [f.values[u:u+div] for u in range(0, f.values.size, div)]
    randchoice = np.zeros((div, f.scope[0].k))
    for i in range(div):
        randchoice[i][np.random.randint(0, f.scope[0].k)] = int(1)
    
    return Factor([v for v in f.scope], list(randchoice.T), f.mod)
    


class ArcInversion:
    
    def __init__(self, influence_diagram):
        self.id = influence_diagram
        cf = self.id.chance_factors
        df = self.id.decision_factors
        uf = self.id.utility_factors
        #uf = Factor([], [0.0], 'utility')
        #for f in self.id.utility_factors:
            #uf += f

        #self.gd = ProbabilityDistribution(cf + [uf] + df)
        self.gd = ProbabilityDistribution(cf + uf + df)
        
        self.gained_utility = []
        self.removed = []
        self.decisions = []
        
    def check_regular_id(self):
        # per costruzione, già dag e diretto
        
        regular, order = find_decision_path(self.id)
        
        if regular == True:
            print('Regular ID')
        else:
            raise Exception('Not regular ID')
            
        g = self.id.nxg()
        # NO FORGET ARCS?  # addddd
        prev = order[0]
        for d in order[1:]:
            info_d = list(set(g.predecessors(d)) & self.id.v_set)
            info_prev = list(set(g.predecessors(prev)) & self.id.v_set)
            comp = list(set(info_prev) - set(info_d))
            if comp:
                for f in self.id.decision_factors:
                    if f.scope[0] == d:
                        f_d = f
                for v in comp:
                    self.id.add_var_to(f_d, v, 'decision')
                        
            prev = d

        return self.id
    
    def update_id(self):
        new_d = []
        new_v = []
        new_u = []
        for factor in self.gd.factors:
            if factor.mod == 'decision':
                new_d.append(factor)
            elif factor.mod == 'utility':
                new_u.append(factor)
            else:
                new_v.append(factor)
                
        self.id = InfluenceDiagram(new_v, new_u, new_d)
        
        return self.id
    
    def barren_node_removal(self, i_d):
        # devo però rimettere le utility per i barren
        g = i_d.nxg()
        factors = self.gd.factors
        nodes = chain(list(i_d.v_set), list(i_d.d_set))
        
        #find
        barren = []
        for node in nodes:

            if g.in_degree(node) == 0 and g.out_degree(node) == 1: # it's a root
                barren.append(node)
            elif g.out_degree(node) == 0 and g.in_degree(node) == 1: # it's a leaf
                barren.append(node)
        #eliminate
        #value node case
        for b in barren:
            #value node case, marginalize simply
            if b in list(i_d.v_set):
                self.remove_chance_node(i_d, b)
                self.removed.append(b)
                #g=g.remove_node(b)
                
            #dec node case, any choice is eq
            if b in list(i_d.d_set):
                for d in list(self.id.decision_factors):
                    if b == d.scope[0]:
                        
                        df = random_decision(d)
                        break
                #normalize?
                self.gd = ProbabilityDistribution(factors + [df])
                #g=g.remove_node(b)
                #upload id
                
        return self.update_id()

    def find_chance_node(self, i_d):
        g = i_d.nxg()
        c_find = None
        for node in list(i_d.v_set):
            if g.out_degree(node) == 1 :
                if str(list(g.successors(node))[0]) in list(i_d.u_set):
                    c_find = node
                
        return  c_find
    
    def remove_chance_node(self, i_d, node):
        #g = i_d.nxg()
        # non serve: rimuovo nodo solo se ha u come unico successore
        #cf = [f for f in self.gd.factors if f.mod == 'chance']
        #factor = reduce(lambda x, y: x * y,
        #             [f for f in cf if node in f.scope[1:]],
        #            Factor([], np.array([1.0])))
        #factor = factor.marginalize(node)

        fc = list(self.gd.factors)

        for u in self.gd.factors:
             if node in u.scope:
                  fc.remove(u)
                  if u.mod == 'chance':
                      x = u
                  if u.mod == 'utility':
                      y = u

        prod = x*y  
        prod.mod = 'utility'   
        fc.append(prod.marginalize(node))
        self.gd.factors = fc
        self.removed.append(node)
        self.gd.factors = self.check_util()
        self.gd.factors = self.check_chance()
        
        self.gd = ProbabilityDistribution(self.gd.factors)
        return self.update_id()
    
                
    def find_decision_node(self, i_d):
        g = i_d.nxg()
        
        d_find = None  
        for node in list(i_d.d_set):
            u_succ = list(set(g.successors(node)) & set(list(i_d.u_set)))
            if u_succ:
                info_d = list(set(g.predecessors(node)) & self.id.v_set)
                for u in u_succ:
                    info_u = list(set(g.predecessors(u)) & self.id.v_set)
                    info_u = list(set(info_u) - set([node]))
                    inter = list(set(info_d) & set(info_u))
                    if inter or len(i_d.decision_factors) == 1:
                        if Counter(inter) == Counter(info_u):
                            d_find = node
                            break
                
                         
        return d_find
                         
            
    def remove_decision_node(self, i_d, node):
        g = i_d.nxg()
        
        for f in self.gd.factors:
            if f.mod == 'decision':
                if f.scope[0] == node:
                    f_d = f
                    
                    break

        eu = ExpectedUtility(i_d)
        #optimal_rule = eu.optimal_decision_rule(f_d.scope, self.decisions)
        optimal_rule = eu.optimal_decision_rule(f_d.scope)
        self.decisions.append(optimal_rule)



        self.removed.append(node)
        self.gd.factors.remove(f_d)
        self.gd.factors.append(Factor(optimal_rule.scope, optimal_rule.values, mod='decision'))
        #best utilty?

        cf = [f for f in self.gd.factors if f.mod == 'chance']
        factor = reduce(lambda x, y: x * y,
                     [f for f in cf if node in f.scope],
                    Factor([], np.array([1.0])))
        factor = factor.marginalize(node)

        #self.gd = self.gd.reduce([(optimal_rule.scope[0], int(np.argwhere(optimal_rule.values == 1)))])
        #self.gd = ProbabilityDistribution(self.gd.factors)
        
        fd = list(self.gd.factors)

        

        for u in self.gd.factors:
             if node in u.scope:
                  fd.remove(u)
                  if u.scope[0] == node and u.mod == 'decision':
                      x = u
                  elif u.mod == 'utility':
                      y = u

                  #else:
                  #if u.mod == 'chance':
                  #  fd.append(u.marginalize(node))
                  #else:
                  #  fd.append(u.marginalize(node))

        prod = x*y  
        prod.mod = 'utility'
        self.gd.factors = self.check_util()
        fd.append(prod.marginalize(node))
        fd.append(factor)
        self.gd.factors = fd
        
        
        self.gd.factors = self.check_chance()
 
        return self.update_id(), optimal_rule

    def check_util(self):
        for factor in self.gd.factors:
            if factor.mod == 'utility':
                inters = list(set(factor.scope) & set(self.removed))
                compl = list(set(factor.scope) - set(inters))
                if not compl:
                    self.gd.factors.remove(factor)
                    self.gained_utility.append(factor)
                elif not factor.scope:
                    self.gd.factors.remove(factor)
                    self.gained_utility.append(factor)
                
                    
                    
        return self.gd.factors
    
    def check_chance(self):
        #for factor in self.gd.factors:
        #    if factor.mod == 'chance':
        #        if len(factor.scope) == 1:
        #            for i, value in enumerate(factor.values):
        #                if value == 1:
        #                    self.gd = self.gd.reduce([(factor.scope[0], i)])
                            
        for factor in self.gd.factors:
            if factor.mod == 'chance':
                if not factor.scope:
                    self.gd.factors.remove(factor)
                    #self.gd = ProbabilityDistribution(self.gd.factors)

              
        return self.gd.factors
        
                         
    def find_node_no_ds(self, i_d):
        g = i_d.nxg()         
        k = None
        for node in list(i_d.v_set):
            print(list(set(g.successors(node))))
            if not list(set(g.successors(node)) & i_d.d_set):
                k = node
                break
            
        return k
                
    
    def reverse_arc(self, i, i_d):
        g = i_d.nxg()
        j = None
        for s in g.successors(i):
            if len(list(nx.all_simple_paths(g, i, s))) == 1:
                j = s
                break
            
        for f in self.gd.factors:
            if f.mod != 'utility':
                if j == f.scope[0]:
                    old_factor_j = f
                elif i == f.scope[0]:
                    old_factor_i = f
        

        new_factor_j = old_factor_j.marginalize(i).to_cpd()
            
        new_factor_i = old_factor_j * old_factor_i / new_factor_j
        
        new_factor_j = new_factor_j
        new_factor_i = new_factor_i.to_cpd()
        self.gd.factors.remove(old_factor_i)
        self.gd.factors.remove(old_factor_j)

        for f in self.gd.factors:
            if f.mod == 'utility':
                if i in f.scope:
                    if j not in f.scope:
                       self.gd.factors.remove(f)
                       self.gd.factors.append(f.__add__(Factor([j])))
                    


        self.gd = ProbabilityDistribution(self.gd.factors + [new_factor_j] + [new_factor_i])
        self.gd.factors = self.check_chance()


                         
        #return new_factor_j, new_factor_i
        return self.update_id()
    
    def solve(self):
                         
        policy = []
                         
        regular = self.check_regular_id()
        tot_ut = len(self.id.utility_factors)
        model=self.barren_node_removal(regular)
        # controlla sta condizione: grafo con utility senza niente?
        print('Start:')
        while len(self.gained_utility) != tot_ut:
            c_to_remove = self.find_chance_node(model)
            print(f'Chance node to remove:', c_to_remove)
            if c_to_remove is not None:
                  model = self.remove_chance_node(model, c_to_remove)
                  print(f'Removed chance node:', c_to_remove)
                  model.viz()
            else:
                d_to_remove = self.find_decision_node(model)
                print(f'Decision node to remove:', d_to_remove)
                if d_to_remove is not None:
                    model, decision = self.remove_decision_node(model, d_to_remove)
                    print(f'Removed decision node:', d_to_remove)
                    policy.append(decision)
                    model.viz()
                    model = self.barren_node_removal(model)
                else:
                    i = self.find_node_no_ds(model)
                    print(f'Node to reverse:', i)
                    while list(set(model.nxg().successors(i)) & model.v_set):
                        model = self.reverse_arc(i, model)
                        print(f'Arc reversed for:', i)
                        model.viz
                    model = self.remove_chance_node(model, i)
                    print(f'Removed node:', i)
                    model.viz()

        return model, policy, self.gained_utility
        
    
    