import { MathHelper } from "../helpers";
import { Organism } from "./organism";


export class Person extends Organism {
  // non-genetic
  name: string;

  // genetic
  speed: number;
  stamina: number;
  health: number;
  weight: number;

  static randomPerson(): Person {
    const p = new Person();
    p.speed = MathHelper.randInt(20, 50);
    p.stamina = MathHelper.randInt(5, 10);
    p.health = MathHelper.randInt(1, 5);
    p.weight = MathHelper.randInt(50, 150);
    return p;
  }

  static zip(person: Person): number[] {
    return [person.speed, person.stamina, person.health, person.weight];
  }

  static unzip(dna: number[]): Person {
    const p = new Person();
    p.speed = dna[0];
    p.stamina = dna[1];
    p.health = dna[2];
    p.weight = dna[3];

    return p;
  }

}