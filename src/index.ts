import { Breeder } from "./genetics/breeder";
import { DecelBreeder } from "./genetics/decel-breeder";
import { FitnessFunction } from "./genetics/delegates";
import { Evolution } from "./genetics/evolution";
import { Person } from "./genetics/person";
import { RNA } from "./genetics/rna";
import { ArrayHelper } from "./helpers";

import breederConfig from "./genetics/breeder-config.json";
import { HiddenStructure, InputStructure, NeuralNetworkStructure, OutputStructure } from "./neural-network/neural-network-structure";
import { NeuralNetworkArray } from "./neural-network/neural-network-array";
import { ActivationFunctions } from "./neural-network/activation-functions";
import { ErrorFunctions } from "./neural-network/error-functions";
import { Matrix } from "./linear-algebra/matrix";
import { minDistOfGraph } from "./algorithms/travelling-salesman";
import { NeuralNetwork } from "./neural-network/neural-network";

function neural() {

  let networkStructure = new NeuralNetworkStructure(
    new InputStructure(1, true), [
      new HiddenStructure(1, true, ActivationFunctions.names.relu)
    ],
    new OutputStructure(1, ActivationFunctions.names.relu),
    ErrorFunctions.names.difference
  )

  let network = new NeuralNetworkArray(networkStructure);

  let inputData = [
    { inputs: [0], expectedOutputs: [0] },
    { inputs: [1], expectedOutputs: [1] }
  ];

  inputData.forEach(data => {
    let result = network.feedForward(data);
    console.log(data, ' -> ', result);
  });

  // let result = network.train(inputData[1]);
  // console.log('Training data: ', inputData[1], '\n\n');
  // console.log('Result: ', result, '\n\n');
  // console.log('Network: ', network);

}

// neural();

function genetic() {
  const rna: RNA = new RNA(Person.unzip, Person.zip);
  // const breeder: Breeder = new Breeder(rna, 0.2, 10);
  const decelBreeder: DecelBreeder = new DecelBreeder(rna, breederConfig.mutateChance, breederConfig.maxMutateAmount,
    breederConfig.generationCount, breederConfig.crossoverChance);

  /*
  speed: number;
  stamina: number;
  health: number;
  weight: number;
  */
  const f: FitnessFunction = (person: Person): number => {
    const weightDelta = 1.5 * Math.abs(120 - person.weight);
    if (rna.zip(person).some(data => data === 0)) {
      return 0;
    }
    return 3 * person.health + 2 * person.stamina + person.speed - weightDelta;
  }

  // const evolution: Evolution = new Evolution(rna, breeder, f);
  const evolution: Evolution = new Evolution(rna, decelBreeder, f);
  let people = ArrayHelper.init(50, () => Person.randomPerson());
  // console.log('people: ', people);

  for (let i = 0; i < breederConfig.generationCount; i++) {
    decelBreeder.setGeneration(i);
    evolution.setOrganisms(people);
    evolution.startEpoch();
    people = (evolution.endEpoch() as Person[]);
    // console.log('people: ', people);
    console.log('avg: ', evolution.avgFitness(people));
    console.log('worst: ', people[0], ' best: ', people[people.length - 1]);

    if (f(people[0]) === f(people[people.length - 1])) {
      console.log('\nConverged at generation ', i);
      break;
    }
  }

  // console.log('\n\npeople: ', people);
}


function matrix() {

  /*
    1 2 3
    4 5 6

    7  8
    9  10
    11 12
  */
    let a: Matrix = new Matrix(2, 3, [[1, 2, 3], [4, 5, 6]]);
    let b: Matrix = new Matrix(3, 2, [[7, 8], [9, 10], [11, 12]]);
    let c: Matrix = new Matrix(2, 3, [[10, 9, 8], [7, 6, 5]]);

    let sum = a.add(c);
    let product = a.multiply(b);

    console.log('a:\n', a.toString());
    console.log('b:\n', b.toString());
    console.log('sum:\n', sum.toString());
    console.log('product:\n', product.toString());

    console.log('transpose:\n', a.transpose());
}

// matrix();

function tsp() {
  let matrix = new Matrix(4, 4, [[0, 2, 5, 6], [2, 0, 10, 7], [5, 10, 0, 4], [6, 7, 4, 0]]);
  console.log(matrix);
  console.log(minDistOfGraph(0, matrix));
}

// tsp();

function nn() {
  let network = new NeuralNetwork();
  network.backpropogate();
}

nn();