import { MathHelper } from "../helpers";
import { Organism } from "./organism";
import { RNA } from "./rna";

export class Breeder {

  rna: RNA;
  mutateChance: number;
  maxMutateAmount: number;
  crossoverChance: number;

  constructor(rna: RNA, mutateChance: number, maxMutateAmount: number, crossoverChance: number=1.0) {

    if (mutateChance >= 1.0) {
      throw RangeError('Mutate chance cannot be >= 1.0');
    }

    this.rna = rna;
    this.mutateChance = mutateChance;
    this.maxMutateAmount = maxMutateAmount;
    this.crossoverChance = crossoverChance;
  }

  /**
   * Create new organism from two organisms using crossover.
   * @param p1 Parent 1.
   * @param p2 Parent 2.
   */
  combine(p1: Organism, p2: Organism, verbose: boolean = false): Organism {
    if (verbose) {
      console.log(`Zipping ${JSON.stringify(p1)} and ${JSON.stringify(p2)}}`);
    }
    const p1DNA = this.rna.zip(p1);
    const p2DNA = this.rna.zip(p2);
    let newDNA: number[] = [];

    // Crossover
    if (Math.random() < this.crossoverChance) {
      if (verbose) {
        console.log('Crossing over DNA');
      }
      const fieldCount = p1DNA.length;
      const crossoverPoint = MathHelper.randInt(0, fieldCount);

      const p1Part = p1DNA.slice(0, crossoverPoint);
      const p2Part = p2DNA.slice(crossoverPoint);
      if (verbose) {
        console.log(`Child gets p1's [${p1Part}] and p2's [${p2Part}]`);
      }
      newDNA = newDNA.concat(p1Part);
      newDNA = newDNA.concat(p2Part);

    } else { // Average
      if (verbose) {
        console.log('Averaging DNA');
      }
      for (let i = 0; i < p1DNA.length; i++) {
        newDNA[i] = (p1DNA[i] + p2DNA[i]) / 2;
      }
    }

    if (verbose) {
      console.log('Mutating DNA');
    }

    const mutatedDNA = this.mutate(newDNA);
    if (verbose) {
      console.log('Unzipping DNA');
    }
    return this.rna.unzip(mutatedDNA);
  }

  /**
   * Apply mutations to the dna.
   * @param dna The dna to mutate.
   */
  mutate(dna: number[]): number[] {

    // Small chance to mutate multiple cells.
    while (Math.random() < this.mutateChance) {
      const up: boolean = Math.random() < 0.5;
      const cellDelta: number = this.maxMutateAmount * Math.random() * (up? 1 : -1);
      const cell = MathHelper.randInt(0, dna.length);
      dna[cell] = Math.max(0, dna[cell] + cellDelta); // Cell data greater than 0
    }

    return dna;
  }
}