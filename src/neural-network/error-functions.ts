import { Matrix } from "../linear-algebra/matrix";

export class ErrorFunctions {

  static names = {
    difference: 'difference'
  }

  static difference(actual: number, expected: number): number {
    return expected - actual;
  }

  static map(name: string): ((actual: number, expected: number) => number) {
    switch (name) {

      default:
        return ErrorFunctions.difference;
    }
  }

  static matrixAverageDifference(actual: Matrix, expected: Matrix): number {
    if (actual.n !== 1 || expected.n !== 1) {
      throw Error(`actual: ${actual}, expected: ${expected}. Dimensions did not match (m, 1)`);
    }
    let result: number = 0;
    for (let i = 0; i < actual.m; i++) {
      result += expected.data[i][0] - actual.data[i][0];
    }
    return result / actual.m;
  }

}