
/*
  Feed forward
    x = input
    ff(x) = fL(WL * fL-1(WL-1 * ... * f1(W1 * x)))
    cost = C(y, ff(x))

    zL - weighted input to each node in layer L
    aL - activation value of each node in layer L

    cache zL for use in (fL)' -> f'(zL)

    dC_wrt_dx = dC_wrt_daL * daL_wrt_dzL * dzL_wrt_daL-1 * daL-1_wrt_dzL-1 * dzL-1_wrt_daL-2 * ... * da1_wrt_dz1
  ->
    dC_wrt_dx = dC_wrt_daL * (fL)' * WL * (fL-1)' * WL-1 * ... * (f1)'
      : d_loss_function, d_activation_functions, matrices of weights

    ^xC = (W1)^T * (f1)' * ... * (WL-1)^T * (fL-1)'

    ol = (f1)' * (Wl+1)^T * ... * (WL-1)^T * (fL-1)'
      : error at level l, gradient of the input values at level l

    ol-1 = (fl-1)' * (Wl)^T * ol
      : recursive formula for gradient at each layer

    ^wlC = ol * (al-1)^T
      : gradients of the weights in layer l :)
*/

import { Matrix } from "../linear-algebra/matrix";
import { ActivationFunctions } from "./activation-functions";
import { ErrorFunctions } from "./error-functions";

export class NeuralNetwork {

  // I -> H -> O

  constructor() {

  }

  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
  static sigmoidDelta(x: number): number {
    return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x));
  }

  backpropogate(): void {
    let inputs: Matrix[] = [
      new Matrix(1, 2, [[0, 0]]),
      new Matrix(1, 2, [[0, 1]]),
      new Matrix(1, 2, [[1, 0]]),
      new Matrix(1, 2, [[1, 1]])
    ];
    let expectedOutputs: Matrix[] = [
      new Matrix(1, 1, [[0]]),
      new Matrix(1, 1, [[0]]),
      new Matrix(1, 1, [[0]]),
      new Matrix(1, 1, [[1]])
    ];

    let ih_weight: Matrix = new Matrix(2, 5, [[0.25, 0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25, 0.25]]);
    let ho_weight: Matrix = new Matrix(5, 1, [[0.25], [0.25], [0.25], [0.25], [0.25]]);
    let margin = 0.001;
    let error: number = 100000;
    let learningRate = 0.001;
    let iteration = 0;
    let maxIterations = 1000000;

/*
    ol = (f1)' * (Wl+1)^T * ... * (WL-1)^T * (fL-1)' * (WL)^T * (fL)' * dCost_wrt_Output
      : error at level l, gradient of the input values at level l

    ol-1 = (fl-1)' * (Wl)^T * ol
      : recursive formula for gradient at each layer

    ^wlC = ol * (al-1)^T
      : gradients of the weights in layer l :)
*/
    let output_a: Matrix;
    let set: number = 0;
    let quitEarly: boolean = false;
    while(Math.abs(error) > margin) {

      // 1x2 * 2x3 -> 1x3
      let hidden_z: Matrix = inputs[set].multiply(ih_weight);
      // 1x3
      let hidden_a: Matrix = hidden_z.map(NeuralNetwork.sigmoid);
      // 1x3 * 3x1 -> 1x1
      let output_z = hidden_a.multiply(ho_weight);
      // 1x1
      output_a = output_z.map(NeuralNetwork.sigmoid);
      // scalar
      error = ErrorFunctions.matrixAverageDifference(output_a, expectedOutputs[set]);
      let dErrorWRTOutput = 1;

      // No weights for output layer.
      // 1x1
      let ooutput = output_a.map(NeuralNetwork.sigmoidDelta).scale(dErrorWRTOutput); // * dCost_wrt_Output which is 1 here.
      // 1x1 = (3x1 -> 1x3) * (1x3 -> 3x1) * 1x1
      let ohidden = ho_weight.transpose().multiply(hidden_z.map(NeuralNetwork.sigmoidDelta).transpose()).multiply(ooutput);
      // 3x1 = 3x1 * 1x1
      let hiddenGradient = hidden_a.transpose().multiply(ohidden);

      // 3x1 = (2x3 -> 3x2) * (1x2 -> 2x1) * 1x1
      let oinput = ih_weight.transpose().multiply(inputs[set].map(NeuralNetwork.sigmoidDelta).transpose()).multiply(ohidden);
      // 2x3 = 3x1 * 1x2
      let inputGradient = inputs[set].transpose().multiply(oinput.transpose());

      // 3x1
      ho_weight = ho_weight.add(hiddenGradient.scale(learningRate));
      // 2x3
      ih_weight = ih_weight.add(inputGradient.scale(learningRate));

      // 0, 1, 2, 3 | 4, 5, 6, 7 | 8, 9, 10, 11
      // 0, 5, 10, 
      if (iteration % Math.floor(inputs.length + 1) === 0) {
        console.log(`input: ${inputs[set]}, output: ${output_a.toString()}, error: ${error}`);
      }

      iteration++;
      if (iteration === maxIterations) {
        quitEarly = true;
        break;
      }
      set = (set + 1) % inputs.length;
    }

    if (!quitEarly) {
      console.log(`\n\nError ${error} < ${margin} after ${iteration + 1} iterations`);
    } else {
      console.log(`\n\nMax iterations reached. Error ${error} after ${iteration + 1} iterations`);
    }
    
    for (set = 0; set < inputs.length; set++) {
      let hidden_z: Matrix = inputs[set].multiply(ih_weight);
      let hidden_a: Matrix = hidden_z.map(NeuralNetwork.sigmoid);
      let output_z = hidden_a.multiply(ho_weight);
      output_a = output_z.map(NeuralNetwork.sigmoid);
      error = ErrorFunctions.matrixAverageDifference(output_a, expectedOutputs[set]);
      console.log(`Expected: ${inputs[set]} -> ${expectedOutputs[set]}\n`);
      console.log(`Actual: ${output_a.toString()}\n`);
      console.log(`Error: ${error}\n\n`);
    }
  }


}