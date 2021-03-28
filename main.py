from models import Function, FunctionParameters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams


def main():
    func = Function()
    fp = FunctionParameters(func)
    best = fp.run()

    print(func)
    print(best)

    fig, ax = plt.subplots()

    target = ax.plot(*func())
    trial, = ax.plot([], [], 'k.')
    time_template = 'generation = %d\nchild = %d\nfitness = %.2f'
    time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)

    dt = 50

    def animate(i):
        g = i # i // fp.population
        f = 0 # i % fp.population
        gen = fp.generation[g]
        fit = fp.fitnesses[g]
        data = [g for _, g in sorted(zip(fit, gen), key=lambda pair: pair[0])][f]
        
        trial.set_data(*data())
        time_text.set_text(time_template % (g, f, fit[f]))
        
        return trial, time_text


    line_ani = animation.FuncAnimation(fig, animate,
                                    range(0, len(fp.generation)),
                                    interval=dt, blit=True, repeat=False)

    plt.show()
    
    
if __name__ = "__main__":
    main()