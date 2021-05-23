'''
Implementation of factor graph and (loopy) belief propagation algorithm.
Current approach (order matters):
-   (1) add RVs
-   (2) add factors to connect them
-   (3) set potentials on factors
-   (4) run inference
-   (5) compute marginals
For some things below, we'll want to represent what's going on in mathematical
notation. Let's define some variables that we'll use throughout to help:
RV vars:
    X       the set of n random variables
    X_i     random variable i (1 <= i <= n)
    v_i     number of values that X_i can take (nonstandard but I wanted one)
    x_ij    a particular value for X_i (1 <= j <= v_i)
    x_i     a simpler (lazy) notation for x_ij (which j doesn't matter)
    x       a set of x_i for i = 1..n (imagine x_1j, x_2k, ..., x_nz)
Factor vars:
    F       the set of m factors
    f_a     factor a (1 <= a <= m) connecting a subset of X
    X_a     the subset of X (RVs) that f_a connects
    x_a     the subset of x (values for RVs) that f_a connects
Functions:
    p(x)    joint distribution for p(X = x)
Notes:
    f_a(x) = f_a(x_a)   Because f_a only touches (is only a function of) x_a,
                        it will "ignore" the other x_i in x that aren't in x_a.
                        Thus, we write f_a(x_a) for convenience to show exactly
                        what f_a operates on.
author: mbforbes
'''
#Fatto dei set potential da riaggiungere
# Imports
# -----------------------------------------------------------------------------
from mypgm.base import RandomVar, Factor, CPD
from mypgm.pgms import ProbabilityDistribution
import itertools



# Builtins
import code  # code.interact(local=dict(globals(), **locals()))
import logging
import signal

DEBUG_DEFAULT = False


# 3rd party
import numpy as np


# Constants
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Settings

# This is the maximum number of iterations that we let loopy belief propagation
# run before cutting it off.
LBP_MAX_ITERS = 50

# Otherwise we'd just make some kinda class to do this anyway.
E_STOP = False


# Functions
# -----------------------------------------------------------------------------

# Let the user Ctrl-C at any time to stop.
def signal_handler(signal, frame):
    logger.info('Ctrl-C pressed; stopping early...')
    global E_STOP
    E_STOP = True
signal.signal(signal.SIGINT, signal_handler)


# Classes
# -----------------------------------------------------------------------------

class MarkovRF:
    '''
    Graph right now has no point, really (except bookkeeping all the RVs and
    factors, assuming we remember to add them), so this might be removed or
    functionality might be stuffed in here later.
    NOTE: All RVs must have unique names.
    TODO: Consider making Node base class which RV and Factor extend.
    TODO: convenience functions or modifications to consider (not worth making
    unless I need them):
        - getters (and setters?) for RVs and Factors
    '''

    def __init__(self, factors, debug=DEBUG_DEFAULT):
        # add now
        self.debug = debug

        # added later
        self._rvs = {}
        # TODO: Consider making dict for speed.
        self._factors = factors
        
        for f in self._factors:
            for v in f.scope:
                if v.name not in self._rvs.keys():
                    self.add_rv(v)

                v.attach(f)

    # TODO(mbforbes): Learn about *args or **args or whatever and see whether I
    #                 can use here to clean this up.

    def has_rv(self, rv_s):
        '''
        Args:
            rv_s (str): Potential name of RV
        Returns:
            bool
        '''
        return rv_s in self._rvs

    def add_rv(self, rv):
        '''
        Args:
            rv (RV)
        '''
        # Check RV with same name not already added.
        if self.debug:
            assert rv.name not in self._rvs
        # Add it.
        self._rvs[rv.name] = rv

    def get_rvs(self):
        '''
        Returns references to actual RVs.
        Returns:
            {str: RV}
        '''
        return self._rvs

    def get_factors(self):
        '''
        Returns references to actual Factors.
        Returns:
            [Factor]
        '''
        return self._factors

    def remove_loner_rvs(self):
        '''
        Removes RVs from the graph that have no factors attached to them.
        Returns:
            int number removed
        '''
        removed = 0
        names = self._rvs.keys()
        for name in names:
            if self._rvs[name].n_edges() == 0:
                self._rvs[name].meta['pruned'] = True
                del self._rvs[name]
                removed += 1
        return removed

    # TODO(mbforbes): Learn about *args or **args or whatever and see whether I
    #                 can use here to clean this up.

    def add_factor(self, factor):
        if self.debug:
            # Check the same factor hasn't already been added.
            assert factor not in self._factors

            # Check factor connecting to exactly the same set of nodes doesn't
            # already exist. This isn't mandated by factor graphs by any means,
            # but it's a heuristic to prevent bugs; if you're adding factors
            # that connect the same set of nodes, you're either doing something
            # weird (and can probably reformulate your graph structure to avoid
            # this duplication), or you have a bug.
            #
            # NOTE(mbforbes): Disabling because I actually do want to be able
            # to do this. Feel free to open GitHub issue for discussion if
            # you're reading this and would like the assert back on.
            #
            # factor_rvs = sorted(factor._rvs)
            # for f in self._factors:
            #     rvs = sorted(f._rvs)
            #     assert factor_rvs != rvs, 'Trying to add factor "%r" but ' \
            #         'factor with the same RVs ("%r") already exists.' % (
            #          factor, f)
        # Add it.
        self._factors += [factor]

    def joint(self, x):
        '''
        Joint is over the factors.
        For a probability, we use the normalization constant 1/Z
            p(x) = 1/Z \product_a^{1..m} f_a(x_a)
        If we don't care what the normalization is, we just write this without
        1/Z:
            p(x) = \product_a^{1..m} f_a(x_a)
        This is currently implemented without normalization. I might want to
        add it in the future. I don't know yet.
        Args:
            x ({str: str|int}) map of node names to assignments. The
                assignments can be labels or indexes
        '''
        # ensure the assignment x given is valid
        if self.debug:
            # check the length (that assignments to all RVs are provided)
            assert len(x) == len(self._rvs)

            # check that each assignment is valid (->)
            for name, label in x.iteritems():
                assert name in self._rvs
                assert self._rvs[name].has_label(label)

            # check that each RV has a valid assignment (<-)
            for name, rv in self._rvs.iteritems():
                assert name in x
                assert rv.has_label(x[name])

        # Do the actual computation.
        # NOTE: This could be sped up as all factors can be computed in
        # parallel.
        prod = 1.0
        for f in self._factors:
            prod *= f.eval(x)
        return prod

    def bf_best_joint(self):
        '''
        Brute-force algorithm to compute the best joint assignment to the
        factor graph given the current potentials in the factors.
        This takes O(v^n) time, where
            v   is the # of possible assignments to each RV
            n   is the # of RVs
        This is bad. This function is given for debugging / proof of concept
        only.
        Returns:
            ({str: int}, float)
        '''
        return self._bf_bj_recurse({}, self._rvs.values())

    def _bf_bj_recurse(self, assigned, todo):
        '''
        Helper method for bf_best_joint.
        Args:
            assigned ({str: int})
            todo ([RV])
        '''
        # base case: just look up the current assignment
        if len(todo) == 0:
            return assigned, self.joint(assigned)

        # recursive case: pull off one RV and try all options.
        best_a, best_r = None, 0.0
        rv = todo[0]
        todo = todo[1:]
        for val in range(rv.n_opts):
            new_a = assigned.copy()
            new_a[rv.name] = val
            full_a, r = self._bf_bj_recurse(new_a, todo)
            if r > best_r:
                best_r = r
                best_a = full_a
        return best_a, best_r

    def lbp(self, init=True, normalize=False, max_iters=LBP_MAX_ITERS,
            progress=False):
        '''
        Loopy belief propagation.
        FAQ:
        -   Q: Do we have do updates in some specific order?
            A: No.
        -   Q: Can we intermix computing messages for Factor and RV nodes?
            A: Yes.
        -   Q: Do we have to make sure we only send messages on an edge once
               messages from all other edges are received?
            A: No. By sorting the nodes, we can kind of approximate this. But
               this constraint is only something that matters if you want to
               converge in 1 iteration on an acyclic graph.
        -   Q: Do factors' potential functions change during (L)BP?
            A: No. Only the messages change.
        '''
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        #
        # note:
        # - every time message sent, normalize if too large or small
        #
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        nodes = self._sorted_nodes()

        # Init if needed. (Don't if e.g. external func is managing iterations)
        if init:
            self.init_messages(nodes)

        cur_iter, converged = 0, False
        while cur_iter < max_iters and not converged and not E_STOP:
            # Bookkeeping
            cur_iter += 1

            if progress:
                # self.print_messages(nodes)
                logger.debug('\titeration %d / %d (max)', cur_iter, max_iters)

            # Comptue outgoing messages:
            converged = True
            for n in nodes:
                n_converged = n.recompute_outgoing(normalize=normalize)
                converged = converged and n_converged

        return cur_iter, converged

    def _sorted_nodes(self):
        '''
        Returns
            [RV|Factor] sorted by # edges
        '''
        rvs = list(self._rvs.values())
        facs = self._factors
        nodes = rvs + facs
        return sorted(nodes, key=lambda x: x.n_edges())

    def init_messages(self, nodes=None):
        '''
        Sets all messages to uniform.
        Args:
            nodes ([RV|Factor], default=None) if None, uses all nodes
        '''
        if nodes is None:
            nodes = self._sorted_nodes()
        for n in nodes:
            n.init_lbp()
            
    def print_sorted_nodes(self):
        print(self._sorted_nodes())

    def print_messages(self, nodes=None):
        '''
        Prints (outgoing) messages for node in nodes.
        Args:
            nodes ([RV|Factor])
        '''
        if nodes is None:
            nodes = self._sorted_nodes()
        print('Current outgoing messages:')
        for n in nodes:
            n.print_messages()

    def rv_marginals(self, rvs=None, normalize=True):
        '''
        Gets marginals for rvs.
        The marginal for RV i is computed as:
            marg = prod_{neighboring f_j} message_{f_j -> i}
        Args:
            rvs ([RV], opt): Displays all if None
            normalize (bool, opt) whether to turn this into a probability
                distribution
        Returns:
            [(RV, np.ndarray)]
        '''
        if rvs is None:
            rvs = self._rvs.values()

        tuples = []
        for rv in rvs:
            # Compute marginal
            name = str(rv)
            marg, _ = rv.get_belief()
            if normalize:
                marg /= sum(marg)

            tuples += [(rv, marg)]
        return tuples

    def print_rv_marginals(self, rvs=None, normalize=True):
        '''
        Displays marginals for rvs.
        The marginal for RV i is computed as:
            marg = prod_{neighboring f_j} message_{f_j -> i}
        Args:
            rvs ([RV], opt): Displays all if None
            normalize (bool, opt) whether to turn this into a probability
                distribution
        '''
        # Preamble
        disp = 'Marginals for RVs'
        if normalize:
            disp += ' (normalized)'
        disp += ':'
        print(disp)

        # Extract
        tuples = self.rv_marginals(rvs, normalize)

        # Display
        for rv, marg in tuples:
            print(str(rv))
            vals = range(rv.k)
            if len(rv.labels) > 0:
                vals = rv.labels
            for i in range(len(vals)):
                print('\t', vals[i], '\t', marg[i])

    def debug_stats(self):
        logger.debug('Graph stats:')
        logger.debug('\t%d RVs', len(self._rvs))
        logger.debug('\t%d factors', len(self._factors))





class Sampler:

    def __init__(self):
        self.samples = []

    def posterior(self, var_assg_pairs):
        var_assg_pairs = list(var_assg_pairs)

        n = len(self.samples)
        if n == 0:
            raise Exception('No samples for estimate')

        m = 0

        for s in self.samples:
            valid = 1

            for var, assg in var_assg_pairs:
                if s[var] != assg:
                    valid = 0
                    break

            m += valid

        return float(m) / n

    def reset(self):
        self.samples = []

    def samples_to_matrix(self):
        if len(self.samples) == 0:
            return None, np.array([], dtype=np.float)

        scope = sorted(self.samples[0].keys())
        var_pos = {v: i for (i, v) in enumerate(scope)}

        X = np.zeros((len(self.samples), len(scope)), dtype=np.int)
        for i, sample in enumerate(self.samples):
            for var, assg in sample.items():
                X[i, var_pos[var]] = assg

        return scope, X

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# summetryc
def uniform_proposal(x, stepsize):
    return np.random.uniform(low=x - 0.5 * stepsize,
                             high=x + 0.5 * stepsize)

def gaussian_proposal(x, sigma):
    """
    Gaussian proposal distribution.

    Draw new parameters from Gaussian distribution with
    mean at current position and standard deviation sigma.

    Since the mean is the current position and the standard
    deviation is fixed. This proposal is symmetric so the ratio
    of proposal densities is 1.

    :param x: Parameter array
    :param sigma:
        Standard deviation of Gaussian distribution. Can be scalar
        or vector of length(x)

    :returns: (new parameters, ratio of proposal densities)
    """

    # Draw x_star
    x_star = x + np.random.randn() * sigma


    return x_star
#acceptance probability of a Metropolis-Hastings sampling step:

def p_acc_MH(x_new, x_old, prob):
    #print(prob.values[prob.scope[0].p[x_new]], prob.values[prob.scope[0].p[x_old]])
    a = prob.values[prob.scope[0].p[x_new]]
    b = prob.values[prob.scope[0].p[x_old]]
    #return min(1, weird_division(prob.values[prob.scope[0].p[x_new]], prob.values[prob.scope[0].p[x_old]]))
    # for the computation is better to use logaritms
    return min(1, np.exp(np.log(a)-np.log(b)))

def sample_MH(x_old, prob, proposal, stepsize):
    mn=prob.scope[0].labels[0]
    mx=prob.scope[0].labels[-1]
    # round for discrete variables
    x_new = np.round(np.clip(proposal(x_old, stepsize),mn,mx))

    # here we determine whether we accept the new state or not:
    # we draw a random number uniformly from [0,1] and compare
    accept = np.random.random() < p_acc_MH(x_new, x_old, prob)
    if accept:
        return accept, x_new
    else:
        return accept, x_old

class GibbsSampler(Sampler):

    def __init__(self, gd, metropolis=False, proposal=None, delta=None):
        super(GibbsSampler, self).__init__()
        self.gd = gd
        self.metropolis = metropolis
        if self.metropolis is True:
            self.accepted = 0
            # proposal can be uniform or gaussian: step size is the std in the case of gaussian
            if proposal is not None:
                self.proposal = proposal
                if delta is not None:
                    self.step_size = delta
                else:
                    self.step_size = 1
            else:
                self.proposal = uniform_proposal        

    def sample(self, evidence=[], var_assg_pairs0=None, burn_in=1000,
               n=1000, plot=None, print_posterior=None):
        gd = self.gd.reduce(evidence)
        variables = gd.variables()
        vars_ = list(gd.variables())

        if var_assg_pairs0:
            s = {v: a for (v, a) in var_assg_pairs0}
        else:
            s = {v: np.random.choice(v.k) for v in variables}

        for t in tqdm(range(burn_in + n)):
            for v in s.keys():
                gd_v = ProbabilityDistribution(
                    [f for f in gd.factors if v in f.scope])
                
                old = s[v]
                del s[v]
                gd_v = gd_v.reduce(s.items())

                f = gd_v.joint()
                # metropolis step if required
                if self.metropolis is True:
   
                    acc, s[v] = sample_MH(old, f, self.proposal, self.step_size)
                    if acc is True:
                        self.accepted += 1
                        
                else:
                    s[v] = f.sample()

            if t >= burn_in:
                self.samples.append(dict(s))
                

        if plot:
            samples = dict()

            for var in list(set(vars_) & set(plot)):
                if var in s.keys():
                    samples[var] = [self.samples[i][var] for i in range(len(self.samples))]
                    #fig, ax = plt.subplots()
                    #ax.hist(samples[var], bins="auto")
                    #ax.axvline(0, color="C1", linestyle="--")
                    #ax.set_title('Hist for {}'.format(var.name))
                    
                    # todo: print also log posterior and Markov Chain
                    
                    fig, ax = plt.subplots()
                    sns.distplot(samples[var], kde=False, norm_hist=True, label='Posterior', ax=ax)
                    ax.axvline(np.array(samples[var]).mean(), ls='--', color='r', label='Mean')
                    ax.set_title('Histogram and KDE for {}'.format(var.name))
                    ax.legend()
                    
                    plot1 = plt.figure()
                    plt.legend(loc='best')
                    plt.grid()
                    plt.plot(range(500),samples[var][-500:], 'bo')
                    plt.title('MCMC for {}'.format(var.name))
                    
                    
                    ax.legend();
                    
        if print_posterior:
            to_print = list(set(vars_) & set(print_posterior))
            
            for c in itertools.product(range(max([v.k for k in to_print])+1), repeat=len(to_print)):
                print('{0}: {1}'.format(c, self.posterior(zip(to_print, c))))
            
                    


import time
from mypgm.exacted import ExpectedUtility

class heuristicID:
    
    def __init__(self, i_d, d_factors):
        self.id = i_d
        self.d_factors = d_factors
        self.eu = ExpectedUtility(self.id)
        self.solution = []
        
    def dimensions(self):
        sing = []
        for f in self.d_factors:
            div = f.values.size // f.scope[0].k
            sing.append(len(f.values[:div]))
                    
        return sing
    
    def scopes(self):
        sc = []
        for f in self.d_factors:
            sc.append(f.scope)
            
        return sc

    def bounds(self, x):
        #feasible = [np.round(i) for i in list(x)]
        feasible = x
        return feasible
    
    def policy(self, x):
        spilt = np.cumsum(self.dimensions())
        vals = np.split(x, spilt)
        
        rules = []
        for v in vals:
            rules.append(np.append(v, np.logical_not(v).astype(int)))
            
        #print(rules)
        
        pol = []
        for s, r in zip(self.scopes(), rules):
            pol.append(Factor(s, r, mod='decision'))
            
        return pol

    def f_obj(self, x):
        
        eu_x = self.eu.expected_utility(self.policy(x))
        return eu_x
    
    def run_algorithm(self, max_epochs, mut, population_size, crossp):
        
        def selection(pop, scores, k=3):
	        # first random selection
	        selection_ix = np.random.randint(len(pop))
	        for ix in np.random.randint(0, len(pop), k-1):
		       # check if better (e.g. perform a tournament)
		         if scores[ix] > scores[selection_ix]:
			          selection_ix = ix
	        return pop[selection_ix]
        
        # crossover two parents to create two children
        def crossover(p1, p2, r_cross):
	    # children are copies of parents by default
	        c1, c2 = p1.copy(), p2.copy()
	        # check for recombination
	        if np.random.rand() < r_cross:
		    # select crossover point that is not on the end of the string
		        pt = np.random.randint(1, len(p1)-2)
		        # perform crossover
		        c1 = p1[:pt] + p2[pt:]
		        c2 = p2[:pt] + p1[pt:]
	        return [c1, c2]
        
        # mutation operator
        def mutation(bitstring, r_mut):
	        for i in range(len(bitstring)):
		     # check for a mutation
		        if np.random.rand() < r_mut:
			        # flip the bit
			        bitstring[i] = 1 - bitstring[i]
        
        dimensions = sum(self.dimensions())
        # hyperparameters
        f_obj = self.f_obj

        errors = []
        # initializazion
        #population = np.random.rand(population_size, dimensions)
        population = [np.random.randint(0, 2, dimensions).tolist() for _ in range(population_size)]
        # evaluation
        
        best_idx, best = 0, population[0]
        
        print(f'Max EU at iteration {0}: {f_obj(population[0])}')
        errors.append(f_obj(population[0]))
        prev_max = f_obj(population[0])

        i=1
        start = time.time()
        print("Starting simulation...")
        while i<max_epochs:
            # evaluate all candidates in the population
            fitness = np.asarray([f_obj(individual) for individual in population])
            # check for new best solution
            best_idx_gen = np.argmax(fitness)
            best_gen = population[best_idx]
            if fitness[best_idx_gen] > prev_max:
                best_idx, best = best_idx_gen, best_gen
                prev_max = fitness[best_idx]
                
            errors.append(prev_max)
            # select parents
            selected = [selection(population, fitness) for _ in range(population_size)]
            # create the next generation
            children = list()
            for j in range(0, population_size, 2):
                # get selected parents in pairs
                p1, p2 = selected[j], selected[j+1]
                # crossover and mutation
                for c in crossover(p1, p2, crossp):
                    # mutation
                    mutation(c, mut)
                    # store for next generation
                    children.append(c)
            
            population = children
            
            i += 1

        print(f'Max EU at the end: {prev_max}')
        end = time.time()
        print(f'End simulation: time spend = {end - start}')
 
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(errors)), errors, 'o-', color = 'r', label = 'EU')
        plt.legend(loc='best')
        plt.title('Expected utility during iterations of the algorithm')
        print(f'Best: {best}')
        self.solution = self.policy(best)
        
        return print(f'Optimal strategy: {self.solution}')
        


