# Use this to turn all debugging on or off. Intended use: keep on when you're
# trying stuff out. Once you know stuff works, turn off for speed. Can also
# specify when creating each instance, but this global switch is provided for
# convenience.

#problemi di attachment
DEBUG_DEFAULT = False


import numpy as np
import operator
import pandas as pd
import inspect
import itertools
from collections import Iterable, Sequence
from itertools import chain, combinations

import inspect 
class ParametricDistribution:
    
    def __init__(self, distribution, f_dict):
        self.distribution = distribution
        self.f_dict = f_dict
        #anche l'ordine è importante
        parameter = inspect.getfullargspec(self.distribution._rvs)[0][1:]
        if not list(set(parameter) & set([str(i) for i in self.f_dict.keys()])):
            raise Exception('Distribution parameters and given keys must correspond')
            
    def get_dist(self, order):
            rvs = []
            dim = [v.k for v in order]
            size = np.prod(dim)
            
            indexes = [np.unravel_index(i, dim) for i in range(size)]
            
            v_val = [[v.label_dict[indexes[j][i]] for i, v in enumerate(order)] for j in range(size)]
            
            for comb in range(size):
                d_args = []
                for b in self.f_dict.keys():
                    j_val = []
                    for w in inspect.getfullargspec(self.f_dict[b])[0]:
                        for i, k in enumerate(order):
            
                            if str(k.name) == str(w):
                
                                j_val.append(v_val[comb][i])
                    
                    d_args.append(self.f_dict[b](*j_val))
                    
                    
                    rvs.append(self.distribution(*d_args))
                
            return rvs


#nome ai valori?
#problema cpd()
#scipy importante
#giustificazione discretization
#aggiusta fatto enumerate
# spiega scrtuuta parametrica per valori continui
# scipy incorporato?
def divide_safezero(a, b):
    '''
    Divies a by b, then turns nans and infs into 0, so all division by 0
    becomes 0.
    Args:
        a (np.ndarray)
        b (np.ndarray|int|float)
    Returns:
        np.ndarray
    '''
    # deal with divide-by-zero: turn x/0 (inf) into 0, and turn 0/0 (nan) into
    # 0.
    c = a / b
    c[c == np.inf] = 0.0
    c = np.nan_to_num(c)
    return c


class RandomVar:

    def __init__(self, name, k=2, inf_sup=None, mod = None, debug=DEBUG_DEFAULT):
        self.name = name
        self.k = k
        self.inf_sup = inf_sup
        self.debug = debug

        if mod is None:
            self.mod = 'chance'
        else:
            self.mod = mod

        # vars added later
        # TODO: consider making dict for speed.
        self._factors = []
        self._outgoing = None

            
        self.labels = self.get_labels()
            
        # anche con index method
        self.label_dict = dict(zip(range(self.k), self.labels))
        self.p = {v: k for k, v in self.label_dict.items()}

        
    def get_labels(self):
        if self.inf_sup != None:
            if self.inf_sup[0] <= self.inf_sup[1]:
                if self.inf_sup[1]+1-self.inf_sup[0] < self.k:
                    raise Exception('Cardinality and extremes does not correspond for a discrete r.v.')
                else:
                    labels = list(int(i) for i in np.linspace(self.inf_sup[0], self.inf_sup[1], self.k))
            else:
                raise Exception('Sup must be greater than inf')
        else: 
            labels = list(range(self.k))
        
        
        return labels

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.name.__hash__()

    def __lt__(self, other):
        return self.name < other.name
    
    
    def get_factors(self):
        '''
        Returns original references
        Returns:
            [Factor]
        '''
        return self._factors

    def get_outgoing(self):
        '''
        Returns COPY
        Returns:
            [np.ndarray]
        '''
        return self._outgoing[:]

    def init_lbp(self):
        '''
        Clears any existing messages and inits all messages to uniform.
        '''
        self._outgoing = [np.ones(self.k) for f in self._factors]

    def print_messages(self):
        '''
        Displays the current outgoing messages for this RV.
        '''
        for i, f in enumerate(self._factors):
            print('\t', self, '->', f, '\t', self._outgoing[i])

    def recompute_outgoing(self, normalize=False):
        '''
        TODO: Consider returning SSE for convergence checking.
        TODO: Is normalizing each outgoing message at the very end the right
              thing to do?
        Returns:
            bool whether this RV converged
        '''
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        # Save old for convergence check.
        old_outgoing = self._outgoing[:]

        # Get all incoming messages.
        total, incoming = self.get_belief()

        # Compute all outgoing messages and return whether convergence
        # happened.
        convg = True
        for i in range(len(self._factors)):
            o = divide_safezero(total, incoming[i])
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                self.k
        return convg

    def get_outgoing_for(self, f):
        '''
        Gets outgoing message for factor f.
        Args:
            f (Factor)
        Returns:
            np.ndarray of length self.n_opts
        '''
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        for i, fac in enumerate(self._factors):
            if f == fac:
                return self._outgoing[i]

    def get_belief(self):
        '''
        Returns the belief (AKA marginal probability) of this RV, using its
        current incoming messages.
        Returns tuple(
            marginal (np.ndarray)   of length self.n_opts         ,
            incoming ([np.ndarray]) message for f in self._factors,
        )
        '''
        incoming = []
        total = np.ones(self.k)
        for i, f in enumerate(self._factors):
            m = f.get_outgoing_for(self)
            if self.debug:
                assert m.shape == (self.k,)
            incoming += [m]
            total *= m
        return (total, incoming)

    def n_edges(self):
        '''
        Returns:
            int how many factors this RV is connected to
        '''
        return len(self._factors)

    def has_label(self, label):
        '''
        Returns whether label indicates a valid value for this RV.
        Args:
            label (int|str)
        returns
            bool
        '''
        # If int, make sure fits in n_opts. If str, make sure it's in the list.
        if len(self.labels) == 0:
            # Tracking ints only. Provided label must be int.
            if self.debug:
                assert type(label) is int
            return label < self.k
        else:
            # Tracking strs only. Provided label can be int or str.
            if self.debug:
                assert type(label) in [int, str]
            if type(label) in [str]:
                return label in self.labels
            # Default: int
            return label < self.k

    def get_int_label(self, label):
        '''
        Returns the integer-valued label for this label. The provided label
        might be an integer (in which case it's already in the correct form and
        will be returned unchanged) or a string (in which case it will be
        turned into an int).
        This assumes the caller has already ensured this is a valid label with
        has_label.
        Args:
            label (int|str)
        returns
            int
        '''
        if type(label) is int:
            return label
        # assume string otherwise
        return self.labels.index(label)

    def attach(self, factor):
        '''
        Don't call this; automatically called by Factor's attach(...). This
        doesn't update the factor's attachment (which is why you shouldn't call
        it).
        factor (Factor)
        '''
        # check whether factor already added to rv; reach factor should be
        # added at most once to an rv.
        if self.debug:
            for f in self._factors:
                # We really only need to worry about the exact instance here,
                # so just using the builtin object (mem address) equals.
                assert f != factor, ('Can\'t re-add factor %r to rv %r' %
                                     (factor, self))

        # Do the registration
        self._factors += [factor]

    
def weird_division(n, d):
    return n / d if d else 0
    
    #aggiusta fatto di values in np
    #incompatibilità dist e values!!!
class Factor:

    """Representation of a factor over discrete random variables"""

    def __init__(self, scope, values=None, mod = None, distribution=None, debug=DEBUG_DEFAULT):
        scope_check = []
        for v in scope:
            if not isinstance(v, RandomVar):
                v = RandomVar(v)
            scope_check.append(v)
            
        self.scope = scope_check
        self.debug = debug
        if mod is None:
            if self.scope:
                 self.mod = self.scope[0].mod
            else:
                self.mod = 'chance'
        else:
            self.mod = mod
        
        #if ( list(values) != None and distribution != None ):
                #raise Exception('The factor parametric distribution is not compatible with predifined values.')
        
        if distribution == None:
            if values is None:
                self.values = np.zeros(np.prod(self.scope_dimensions()))
            else:
                # spiega questo: conversione 
                if isinstance(values[0], Iterable):
                    self.values = np.array([item for sublist in values for item in sublist])
                    if self.values.size != np.prod(self.scope_dimensions()):
                        raise Exception('Incorrect table given scope')      
                elif len(list(values)) != np.prod(self.scope_dimensions()):
                    raise Exception('Incorrect table given scope')

                else:
                    self.values = np.array(values)
                    #self.set_potential(self.get_sub_vals())

        else:
            if self.mod == 'decision':
                raise Exception('The factor parametric distribution is not compatible with decision nodes.')
            else:
                rv = self.scope[0]
                if len(self.scope) == 1:
                    self.distribution = distribution
                    self.values = [self.distribution.pmf(i) for i in rv.labels]
                    #ottieni tavola da pmf e labels da var
                    #discretizza da pdf e labels da var
                
                else:
                    
                    if not isinstance(distribution, ParametricDistribution):
                        raise Exception('The distribution must be parametric for multi-variables factors.')
                    else:
                        self.distribution = distribution.get_dist(self.scope[1:])
                        self.values = [self.distribution[j].pmf(i) for i in rv.labels for j in range(len(self.distribution))]
                        self.values = np.array(self.values)
                    #self.values = [self.distribution.pmf(self.itoa(i)) for i in range(np.prod(self.scope_dimensions()))]:
                    #servono funzioni parametriche 
                    #le variabili in scope[1:] dovrebbero assumere solo valori noti
                    #inseriscili nella multivar e nel caso discretizza
                    
        self._potential = None
        self._outgoing = None
        self.debug = debug


    def n_edges(self):
        '''
        Returns:
            int how many RVs this Factor is connected to (ma non è n edges..)
        '''
        return len(self.scope)

    def init_lbp(self):
        '''
        Clears any existing messages and inits all messages to uniform.
        '''
        self._outgoing = [np.ones(r.k) for r in self.scope]

    def get_outgoing(self):
        '''
        Returns COPY of outgoing.
        Returns:
            [np.ndarray] where element i is of shape get_rvs()[i].n_opts
        '''
        return self._outgoing[:]

    def get_outgoing_for(self, rv):
        '''
        Gets the message for the random variable rv that this factor is
        connected to. Prereq: this must be connected to rv. Duh. This code
        doesn't check that.
        Args:
            rv (RV)
        Returns:
            np.ndarray of length rv.n_opts
        '''
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        for i, r in enumerate(self.scope):
            if r == rv:
                return self._outgoing[i]

    def recompute_outgoing(self, normalize=False):
        '''
        TODO: Consider returning SSE for convergence checking.
        Returns:
            bool whether this Factor converged
        '''
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        # Save old for convergence check.
        old_outgoing = self._outgoing[:]

        # (Product:) Multiply RV messages into "belief".
        incoming = []
        belief = self.get_sub_vals().copy()
        #belief = self.values.copy()
        for i, rv in enumerate(self.scope):
            m = rv.get_outgoing_for(self)
            if self.debug:
                assert m.shape == (rv.k,)
            # Reshape into the correct axis (for combining). For example, if
            # our incoming message (And thus rv.n_opts) has length 3, our
            # belief has 5 dimensions, and this is the 2nd (of 5) dimension(s),
            # then we want the shape of our message to be (1, 3, 1, 1, 1),
            # which means we'll use [1, -1, 1, 1, 1] to project our (3,1) array
            # into the correct dimension.
            #
            # Thanks to stackoverflow:
            # https://stackoverflow.com/questions/30031828/multiply-numpy-
            #     ndarray-with-1d-array-along-a-given-axis
            proj = np.ones(len(belief.shape), int)
            proj[i] = -1
            m_proj = m.reshape(proj)
            incoming += [m_proj]
            # Combine to save as we go
            belief *= m_proj

        # Divide out individual belief and (Sum:) add for marginal.
        convg = True
        all_idx = range(len(belief.shape))
        for i, rv in enumerate(self.scope):
            rv_belief = divide_safezero(belief, incoming[i])
            axes = tuple(list(all_idx[:i]) + list(all_idx[i+1:]))
            o = rv_belief.sum(axis=axes)
            if self.debug:
                assert self._outgoing[i].shape == (rv.k, )
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                rv.k

        return convg

    def print_messages(self):
        '''
        Displays the current outgoing messages for this Factor.
        '''
        for i, rv in enumerate(self.scope):
            print('\t', self, '->', rv, '\t', self._outgoing[i])

    def attach(self, rv):
        '''
        Call this to attach this factor to the RV rv. Clears any potential that
        has been set.
        rv (RV)
        '''
        # check whether rv already added to factor; reach rv should be added at
        # most once to a factor.
        if self.debug:
            for r in self.scope:
                # We really only need to worry about the exact instance here,
                # so just using the builtin object (mem address) equals.
                assert r != rv, 'Can\'t re-add RV %r to factor %r' % (rv, self)

        # register with rv
        rv.attach(self)

        # Clear potential as dimensions no longer match.
        #self._potential = None            
            
    def eval(self, x):
        '''
        Returns a single cell of the potential function.
        Call this factor f_a. This returns f_a's potential function value for a
        full assignment to X_a, which we call x_a.
        Note that we accept x passed in, which is a full assignment to x. This
        accepts either x (full assignment) or x_a (assignment that this factor
        needs). This function only uses x_a [subset of] x.
        This checks (if debug is on) that all attached RVs have a valid
        assignment in x. Note that if this is begin called from Graph.joint(),
        this property is also checked there.
        '''
        if self.debug:
            # check that each RV has a valid assignment (<-)
            for rv in self.scope:
                assert rv.name in x
                assert rv.has_label(x[rv.name])

        # slim down potential into desired value.
        ret = self.values
        for r in self.scope:
            ret = ret[r.get_int_label(x[r.name])]

        # should return a single number
        if self.debug:
            assert type(ret) is not np.ndarray

        return ret
    
    def scope_dimensions(self):
        return [v.k for v in self.scope]

    def itoa(self, i):
        return np.unravel_index(i, self.scope_dimensions())

    def atoi(self, assg):
        return np.ravel_multi_index(assg, self.scope_dimensions())

    def operation(self, oper, factor):
        """Perform a binary operation `oper` between this factor and `factor`.

        This method generalizes factor products to other binary operations.

        Parameters
        ----------
        oper : callable
            Binary operator on real numbers.
        factor : Factor
            Factor whose operation we want to compute with `self`.

        Returns
        -------
        Factor
            Factor corresponding to the binary operation between `self` and
            `factor`.

        """
        if len(factor.scope) == 0:
            return Factor(list(self.scope),
                          oper(self.values, factor.values[0]))
        if len(self.scope) == 0:
            return Factor(list(factor.scope),
                          oper(factor.values, self.values[0]))

        scope_both = [v for v in self.scope if v in factor.scope]
        scope_self = [v for v in self.scope if v not in scope_both]
        scope_factor = [v for v in factor.scope if v not in scope_both]

        scope = scope_self + scope_both + scope_factor
        if self.mod == 'utility': 
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])), mod=self.mod)
        else:
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))

        assg_self = [output.scope.index(v) for v in self.scope]
        assg_f = [output.scope.index(v) for v in factor.scope]

        for i in range(output.values.size):
            assg = np.array(output.itoa(i))

            i_self = self.atoi(assg[assg_self])
            i_f = factor.atoi(assg[assg_f])

            output.values[i] = oper(self.values[i_self], factor.values[i_f])

        return output

    def __mul__(self, factor):
        return self.operation(operator.mul, factor)
    
    def __truediv__(self, factor):
        return self.operation(weird_division, factor)

    def __add__(self, factor):
        return self.operation(operator.add, factor)

    def marginalize(self, var):
        if not isinstance(var, RandomVar):
                var = RandomVar(var)
        scope = [v for v in self.scope if v != var]
        if len(scope) == 0:
            return Factor([], np.array([sum(self.values)]))
        
        if self.mod == 'utility': 
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])), mod=self.mod)
        else:
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))

        assg_marg = [self.scope.index(v) for v in scope]
        for i in range(self.values.size):
            j = output.atoi(np.array(self.itoa(i))[assg_marg])

            output.values[j] += self.values[i]

        return output

    def maximize(self, var):
        if not isinstance(var, RandomVar):
            var = RandomVar(var)
        scope = [v for v in self.scope if v != var]
        if len(scope) == 0:
            return Factor([], np.array([max(self.values)]))

        if self.mod == 'utility': 
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])), mod=self.mod)
        else:
            output = Factor(scope, np.zeros(np.prod([v.k for v in scope])))
        output.values.fill(-np.infty)

        assg_marg = [self.scope.index(v) for v in scope]
        for i in range(self.values.size):
            j = output.atoi(np.array(self.itoa(i))[assg_marg])

            if self.values[i] > output.values[j]:
                output.values[j] = self.values[i]

        return output

    def argmax(self, partial_assg):
        assg = np.array(partial_assg, dtype=int)
        ind = np.where(assg == -1)[0][0]

        assg[ind] = 0

        maximum = float('-inf')
        imaximum = 0
        for i in range(self.scope[ind].k):
            assg[ind] = i
            if self.values[self.atoi(assg)] > maximum:
                imaximum = i
                maximum = self.values[self.atoi(assg)]

        return imaximum

    def observe(self, evidence):
        if self.mod == 'utility': 
            j = Factor(list(self.scope), np.array(self.values), mod=self.mod)
        else:
            j = Factor(list(self.scope), np.array(self.values))

        for var, value in evidence:
            var_index = self.scope.index(var)

            for i in range(self.values.size):
                assg = self.itoa(i)[var_index]
                if assg != value:
                    j.values[i] = 0

        return j

    def reduce(self, evidence):
        j = self.observe(evidence)

        for var, _ in evidence:
            j = j.marginalize(var)

        return j

    def sample(self, evidence=[]):
        j = self.reduce(evidence)

        if len(j.scope) != 1:
            raise Exception('Invalid scope for sampling')

        return np.random.choice(range(j.scope[0].k), p=j.values)

    def normalize(self):
        return Factor(list(self.scope), self.values / self.values.sum())

    def to_cpd(self):
        if len(self.scope) == 0:
            return CPD([], np.array([1.0]))
        elif len(self.scope) == 1:
            value = self.values.sum()
            if np.allclose(value, 0):
                k = self.scope[0].k
                return CPD(list(self.scope), np.ones(k, dtype=np.float) / k)
            else:
                return CPD(list(self.scope), self.values / value)

        values = np.array(self.values)

        marginal = self.marginalize(self.scope[0])
        for i in range(len(values)):
            value = marginal.values[marginal.atoi(self.itoa(i)[1:])]

            if np.allclose(value, 0):
                values[i] = 1. / self.scope[0].k
            else:
                values[i] = values[i] / value

        return CPD(list(self.scope), values)

    def add_scalar(self, scalar):
        return Factor(list(self.scope), self.values + scalar)

    def __repr__(self):
        if len(self.scope) == 0:
            return '{0}\n'.format(self.values[0])

        string = ''.join([v.name + ' ' for v in self.scope]) + '\n'
        for i in range(self.values.size):
            string += '{0} -> {1}\n'.format(self.itoa(i), self.values[i])
        return string

    def __eq__(self, other):
        return self.scope == other.scope and \
            np.allclose(self.values, other.values)
    
    def get_sub_vals(self):
        # div = self.values.size // self.scope[0].k
        if len(self.scope) != 1:

                
            data = self.values.reshape(([self.scope[i].k for i in range(len(self.scope))]))
        else:
            data = self.values
            
        return data
                
    
    def to_dataframe(self):
        div = self.values.size // self.scope[0].k
        data = [self.values[u:u+div] for u in range(0, self.values.size, div)]
        index = []
        for i in range(self.scope[0].k):
            index.append(self.scope[0].label_dict[i])
            
        columns = []
        
        for i in range(div):
            pos=1
            arrays = []
            for v in self.scope[1:]:
                arrays.append(str(v.name) + '=' + str(v.label_dict[self.itoa(i)[pos]]))
                pos += 1
            s = ','.join(arrays)
            columns.append(s)
            
        cpt = pd.DataFrame(data, index=index, columns=columns)
        cpt.index.name = str(self.scope[0])
        
        return cpt


class CPD(Factor):

    def __init__(self, scope, values=None, mod=None, distribution=None, debug=DEBUG_DEFAULT,):
        Factor.__init__(self, scope, values, mod, distribution, debug=DEBUG_DEFAULT,)
        
        if not self.valid():
            raise Exception('Invalid CPD')

    def valid(self):
        
        if not np.allclose((self.values >= 0), 1):
            return False

        if len(self.scope) == 1:
            
            return np.allclose(sum(self.values), 1.0)
        
        
        rscope = [v.k for v in self.scope[1:]]
        acc = np.zeros(np.prod(rscope))
        
        #sum here over the dependent variables
        for i in range(self.values.size):
            j = np.ravel_multi_index(self.itoa(i)[1:], rscope)
            acc[j] += self.values[i]

        return np.allclose(acc, 1.0)