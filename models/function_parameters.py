from lib import GeneticAlgorithm

import numpy as np

class Function:
    
    def __init__(self, a=None, b=None, c=None, d=None, 
                 xmin=None, xmax=None, density=None):
        self.params = a, b, c, d
        # defaut domain
        self.xmin = xmin if not xmin is None else np.random.uniform()
        self.xmax = xmax if not xmax is None else np.random.uniform(2, 100)
        self.density = density if not density is None else 10
        self.x = np.linspace(self.xmin, self.xmax, self.points)
    
    @property
    def params(self):
        return self.a, self.b, self.c, self.d
    
    @params.setter
    def params(self, *params):
        if len(params[0]) != 4:
            print(params)
            raise Exception(f'Received {len(params)} values; expected exactly 4.')
        a, b, c, d = params[0]
        self.a = a if not a is None else np.random.uniform(0, 10)
        self.b = b if not b is None else np.random.uniform(0, 2 * np.pi)
        self.c = c if not c is None else np.random.uniform(-1, 1)
        self.d = d if not d is None else np.random.uniform(-1, 1)
    
    @property
    def defaults(self):
        return {
            'xmin': self.xmin,
            'xmax': self.xmax,
            'density': self.density
        }
    
    @property
    def domain(self):
        return self.xmin, self.xmax
    
    @domain.setter
    def domain(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    
    @property
    def points(self):
        return int(self.xmax - self.xmin) * self.density
    
    @points.setter
    def points(self, _):
        raise Exception('Do not set this variable directly; try setting either xmin, xmax or density.')
    
    def __str__(self):
        return f"{self.a:.4f} * cos({self.b:.4f} * x) * exp({self.c:.4f} * (x - {self.d:.4f}))"
    
    def __unicode__(self):
        return f"{self.a:.4f} * cos({self.b:.4f} * x) * exp({self.c:.4f} * (x - {self.d:.4f}))"
    
    def __call__(self, x=None):
        if not x:
            x = self.x
        x = np.array(x)
        return x, self.a * np.cos(self.b * x) * np.exp(self.c * np.subtract(x, self.d))
    
class FunctionParameters(GeneticAlgorithm):
    
    
    def __init__(self, func, population=50, generations=1000,
                 keep=.4, mutation=.4, crossover=.15):
        self.func = func
        self.population = population
        self.generations = generations
        self.keep = keep
        self.mutation = mutation
        self.crossover_prob = crossover
        self.generation = {}
        self.fitnesses = {}
    
    async def generate(self):
        return self.func.__class__(**self.func.defaults)
    
    async def fitness(self, gene):
        return np.average(np.power(self.func()[1] - gene()[1], 2))
    
    async def crossover(self, gene1, gene2):
        params = [np.random.choice(vals, p=[self.crossover_prob, 1 - self.crossover_prob])
                  for vals in zip(gene1.params, gene2.params)]
        return self.func.__class__(*params, **self.func.defaults)
    
    async def mutate(self, gene):
        params = [param * np.random.uniform(.9, 1.1)
                  if np.random.uniform() < self.mutation
                  else param
                  for param in gene.params]
        gene.params = params
        return gene