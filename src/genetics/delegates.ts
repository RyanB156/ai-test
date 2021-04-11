import { Organism } from "./organism";

/**
 * Decomposes an organism into its DNA
 */
export type Zipper = (organism: Organism) => number[];
/**
 * Reconstitutes an organism from DNA
 */
export type Unzipper = (dna: number[]) => Organism;
/**
 * Judges organisms to determine survival of the fittestnp
 */
export type FitnessFunction = (organism: Organism) => number;