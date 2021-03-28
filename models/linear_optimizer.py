from lib import GeneticAlgorithm

class LinearOptimizer(GeneticAlgorithm):
    
    
    def __init__(self, costs, constraint, limit, maximize=True):
        self.costs = costs
        self.constraint = constraint
        self.limit = limit
        self.maximize = maximize
        
    @property
    def population(self):
        pass
    
    @property
    def generations(self):
        pass
    
    @property
    def current_generation(self):
        pass
    
    async def generate(self):
        pass
    
    async def crossover(self, gen1, gen2):
        pass
    
    async def mutate(self, gen, probability):
        pass
    
    async def fit(self, gen):
        pass