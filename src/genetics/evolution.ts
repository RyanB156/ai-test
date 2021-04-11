import { FitnessFunction } from "./delegates";
import { Organism } from "./organism";
import { RNA } from "./rna";
import { ArrayHelper } from "../helpers";
import { Breeder } from "./breeder";

export class Evolution {
  evolver: RNA;
  organisms: Organism[];
  fitnessFunction: FitnessFunction;
  breeder: Breeder;
  verbose: boolean;
  fitnessMapping: Map<number, number>;

  constructor(evolver: RNA, breeder: Breeder, fitnessFunction: FitnessFunction, verbose: boolean = false) {
    this.evolver = evolver;
    this.breeder = breeder;
    this.fitnessFunction = fitnessFunction;
    this.verbose = verbose;
  }

  startEpoch(): void {
    if (this.verbose) {
      console.log('Starting evolution');
    }
    this.fitnessMapping = new Map<number, number>();
  }

  endEpoch(): Organism[] {
    this.judgeFitness();
    this.cull();
    this.breed();
    return this.organisms;
  }

  /**
   * Determine the fitness of each organism based on the fitness function.
   */
  judgeFitness(): void {
    if (this.verbose) {
      console.log('Judging organisms...');
    }
    for (let i = 0; i < this.organisms.length; i++) {
      const o: Organism = this.organisms[i];
      const fitness = this.fitnessFunction(o);
      if (this.verbose) {
        console.log(`Organism ${i}: ${fitness}`);
      }
      this.fitnessMapping.set(i, fitness);
    }

    const avg = this.avgFitness(this.organisms);

    this.organisms.sort((a, b) => {
      const af = this.fitnessFunction(a);
      const bf = this.fitnessFunction(b);
      return af - bf;
    })
    if (this.verbose) {
      console.log(`Best organism: ${JSON.stringify(this.organisms[this.organisms.length - 1])}`);
      console.log(`Worst organism: ${JSON.stringify(this.organisms[0])}`);

      console.log('Judging complete');
    }
  }

  avgFitness(organisms: Organism[]): number {
    const averageFitness = ArrayHelper.avg(Array.from(this.fitnessMapping.values()));
    if (this.verbose) {
      console.log(`Average fitness: ${averageFitness}`);
    }
    return averageFitness;
  }

  /**
   * Remove the bottom 50% of organisms.
   */
  cull(): void {
    if (this.verbose) {
      console.log('Culling organisms...');
    }

    // Keep the best 50% of organisms.
    this.organisms = this.organisms.slice(this.organisms.length / 2);
    if (this.verbose) {
      console.log(`${this.organisms.length} organisms remaining`)
    }
    if (this.verbose) {
      console.log('Culling complete');
    }
  }

  /**
   * Restore the 50% of organisms that were lost by creating new children and adding them to the population.
   */
  breed(): void {
    if (this.verbose) {
      console.log('Breeding organisms...');
    }
    const children: Organism[] = [];
    for (let i = 0; i < this.organisms.length - 1; i++) {
      const p1 = this.organisms[i];
      const p2 = this.organisms[i + 1];
      children.push(this.breeder.combine(p1, p2, this.verbose));
    }
    children.push(this.breeder.combine(this.organisms[0], this.organisms[this.organisms.length - 1], this.verbose));
    this.organisms = this.organisms.concat(children);
    if (this.verbose) {
      console.log('Breeding complete');
    }
  }

  setOrganisms(organisms: Organism[]): void {
    if (this.verbose) {
      console.log(`Setting ${organisms.length} like ${JSON.stringify(organisms[0])}`);
    }
    this.organisms = organisms;
  }

  newOrganisms(organismCount: number, spawnerF: (id: number) => Organism) {
    if (this.verbose) {
      console.log(`Setting ${organismCount} new organisms based on the spawner function.`);
    }
    this.organisms = [...Evolution.organismSpawner(organismCount, spawnerF)];
  }

  private static* organismSpawner(count: number, spawnerF: (id: number) => Organism) {
    for (let i = 0; i < count; i++) {
      yield spawnerF(i);
    }
  }

}