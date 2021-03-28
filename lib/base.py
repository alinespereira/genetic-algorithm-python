from abc import ABC
from abc import (abstractmethod,
                 abstractproperty,
                 abstractclassmethod,
                 abstractstaticmethod)

import asyncio
import numpy as np

class GeneticAlgorithm(ABC):
    
    @property
    def generations(self):
        return self._generations
    
    @generations.setter
    def generations(self, gen):
        self._generations = gen

    @property
    def population(self):
        return self._population
    
    @population.setter
    def population(self, pop):
        self._population = pop
    
    @property
    def current_generation(self):
        return self.generation[max(self.generation.keys())]
    
    @property
    def current_fitnesses(self):
        if max(self.generation.keys()) in self.fitnesses:
            return self.fitnesses[max(self.generation.keys())]
        else:
            self.fitnesses[max(self.generation.keys())] = []
            return self.fitnesses[max(self.generation.keys())]
    
    @current_fitnesses.setter
    def current_fitnesses(self, values):
        self.fitnesses[max(self.generation.keys())] = values
    
    @property
    def keep(self):
        return int(self._keep * self._population)
    
    @keep.setter
    def keep(self, keep):
        self._keep = keep
    
    @abstractmethod
    def generate(self): pass
    
    async def initialize_population(self):
        return await asyncio.gather(*(self.generate()
                                      for _ in range(self.population)))
                                    
    @abstractmethod
    async def fitness(self, gene): pass
    
    async def select(self):
        fitness = await asyncio.gather(*(self.fitness(g)
                                         for g in self.current_generation))
        self.current_fitnesses = fitness
        best = [g for _, g in sorted(zip(self.current_fitnesses,
                                         self.current_generation),
                                     key=lambda pair: pair[0])]
        return best[:self.keep]
    
    @abstractmethod
    async def crossover(self, gene1, gene2): pass
    
    @abstractmethod
    async def mutate(self, gene): pass
    
    async def fit(self):
        self.generation[0] = await self.initialize_population()
        self.fitnesses[0] = []
        for i in range(self.generations):
            best = await self.select()
            offspring = []
            for _ in range(len(best), self.population):
                gene1, gene2 = np.random.choice(best, size=2)
                child = await self.crossover(gene1, gene2)
                offspring.append(child)
            offspring = await asyncio.gather(*(self.mutate(gene)
                                               for gene in offspring))
            self.generation[i + 1] = best + offspring
        
        best = await self.select()
        return best[0]
    
    def run(self):
        return asyncio.run(self.fit())