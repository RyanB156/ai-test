import { Breeder } from "./breeder";
import { RNA } from "./rna";

// Breeder that decreases the mutation rate with each generation
export class DecelBreeder extends Breeder {

  generationCount: number;

  baseMutateChance: number;
  mutateChanceDelta: number;

  baseMaxMutateAmount: number;
  maxMutateAmountDelta: number;

  constructor(rna: RNA, mutateChance: number, maxMutateAmount: number, generationCount: number, crossOverChance: number=1.0) {
    super(rna, mutateChance, maxMutateAmount, crossOverChance);
    this.generationCount = generationCount;

    this.baseMutateChance = mutateChance;
    this.mutateChanceDelta = this.baseMutateChance / generationCount;

    this.baseMaxMutateAmount = maxMutateAmount;
    this.maxMutateAmountDelta = this.baseMaxMutateAmount / generationCount;
  }

  /**
   * Set the generation count before each cycle to slow mutation as the generations pass.
   * @param gen The current generation number. 0 behaves like normal breeder.
   */
  setGeneration(gen: number): void {
    this.mutateChance = this.baseMutateChance - (gen * this.mutateChanceDelta);
    this.maxMutateAmount = this.baseMaxMutateAmount - (gen * this.maxMutateAmountDelta);
  }
}