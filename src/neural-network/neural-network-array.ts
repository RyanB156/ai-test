import { ArrayHelper, MathHelper } from "../helpers";
import { ActivationFunctions } from "./activation-functions";
import { ErrorFunctions } from "./error-functions";
import { NeuralNetworkStructure } from "./neural-network-structure";


export class InputLayer {
  count: number;
  weights: number[][];
  weightDeltas: number[][];
  biasWeights: number[];
  inputData: number[];

  constructor(count: number, nextLayerCount: number, biasEnabled: boolean=true) {
    this.count = count;
    this.weights = ArrayHelper.init(count, _ => ArrayHelper.init(nextLayerCount, _ => Math.random()));
    this.weightDeltas = ArrayHelper.init(count, _ => ArrayHelper.init(nextLayerCount, _ => 0));
    if (biasEnabled) {
      this.biasWeights = ArrayHelper.init(nextLayerCount, _ => Math.random());
      this.weightDeltas.push(ArrayHelper.init(nextLayerCount, _ => 0));
    } else {
      this.biasWeights = undefined;
    }
  }
}

export class HiddenLayer {
  count: number;
  weights: number[][];
  weightDeltas: number[][];
  activationFunction: (input: number) => number;
  deltaActivationFunction: (input: number) => number;
  biasWeights: number[]

  inputData: number[];
  outputData: number[];

  constructor(count: number, nextLayerCount: number, activationFunctionName: string, biasEnabled: boolean=true) {
    this.count = count;
    this.weights = ArrayHelper.init(count, _ => ArrayHelper.init(nextLayerCount, _ => Math.random()));
    this.weightDeltas = ArrayHelper.init(count, _ => ArrayHelper.init(nextLayerCount, _ => 0));
    [this.activationFunction, this.deltaActivationFunction] = ActivationFunctions.map(activationFunctionName);

    if (biasEnabled) {
      this.biasWeights = ArrayHelper.init(nextLayerCount, _ => Math.random());
      this.weightDeltas.push(ArrayHelper.init(nextLayerCount, _ => 0));
    } else {
      this.biasWeights = undefined;
    }

    this.inputData = ArrayHelper.init(this.count, _ => 0);
    this.outputData = ArrayHelper.init(this.count, _ => 0);
  }

  clearInputAndOutputData(): void {
    this.inputData = ArrayHelper.init(this.count, _ => 0);
    this.outputData = ArrayHelper.init(this.count, _ => 0);
  }

}

export class OutputLayer {
  count: number;
  activationFunction: (input: number) => number;
  deltaActivationFunction: (input: number) => number;

  inputData: number[];
  outputData: number[];

  constructor(count: number, activationFunctionName: string) {
    this.count = count;
    [this.activationFunction, this.deltaActivationFunction] = ActivationFunctions.map(activationFunctionName);
    this.inputData = ArrayHelper.init(this.count, _ => 0);
    this.outputData = ArrayHelper.init(this.count, _ => 0);
  }
}

export class NeuralNetworkArray {

  private inputLayer: InputLayer;
  private hiddenLayers: HiddenLayer[];
  private outputLayer: OutputLayer;
  private errorCalculator: (actual: number, expected: number) => number;

  constructor(networkStructure: NeuralNetworkStructure) {
    // Create input layer.
    let firstHiddenLayerCount = networkStructure.hiddenLayers[0].count;
    this.inputLayer = new InputLayer(networkStructure.inputLayer.count, firstHiddenLayerCount,
      networkStructure.inputLayer.biasEnabled);

    // Create hidden layer.
    this.hiddenLayers = [];
    let hiddenLayer = 0;
    for (; hiddenLayer < networkStructure.hiddenLayers.length - 1; hiddenLayer++) {
      let currentLayerNodeCount = networkStructure.hiddenLayers[hiddenLayer].count;
      let nextLayerNodeCount = networkStructure.hiddenLayers[hiddenLayer + 1].count;
      let activationFunctionName = networkStructure.hiddenLayers[hiddenLayer].activationFunctionName;

      this.hiddenLayers.push(new HiddenLayer(currentLayerNodeCount, nextLayerNodeCount, activationFunctionName));
    }

    let currentLayerNodeCount = networkStructure.hiddenLayers[hiddenLayer].count;
    let outputLayerCount = networkStructure.outputLayer.count;
    let activationFunctionName = networkStructure.hiddenLayers[hiddenLayer].activationFunctionName;
    this.hiddenLayers.push(new HiddenLayer(currentLayerNodeCount, outputLayerCount, activationFunctionName));

    // Create output layer.
    this.outputLayer = new OutputLayer(networkStructure.outputLayer.count, networkStructure.outputLayer.activationFunctionName);

    // Create error function.
    this.errorCalculator = ErrorFunctions.map(networkStructure.errorFunctionName);
  }

  /**
   * Run data through the network and get the result and error.
   * Expects the input data length to match the number of input nodes.
   * Expects the expected output data length to match the number of output nodes.
   * @param data Inputs to the network and the expected outputs.
   * @returns An object containing the output and error for each output node.
   */
  public feedForward(data: {inputs: number[], expectedOutputs: number[]}): {result: number[], error: number[]} {

    if (data.inputs.length !== this.inputLayer.count) {
      throw Error('Input data did not match the number of input nodes');
    }

    if (data.expectedOutputs.length !== this.outputLayer.count) {
      throw Error('Output data did not match the number of output nodes');
    }

    this.inputLayer.inputData = data.inputs;

    // Loop over each hidden node in the first hidden layer.
    for (let hiddenNode = 0; hiddenNode < this.hiddenLayers[0].count; hiddenNode++) {
      this.hiddenLayers[0].clearInputAndOutputData();
      // Add bias to the input for this node in hidden layer 0.
      if (this.inputLayer.biasWeights !== undefined) {
        this.hiddenLayers[0].inputData[hiddenNode] += this.inputLayer.biasWeights[hiddenNode];
      }
      // Loop over each input node in the input layer.
      for (let inputNode = 0; inputNode < this.inputLayer.count; inputNode++) {
        this.hiddenLayers[0].inputData[hiddenNode] += data.inputs[inputNode] * this.inputLayer.weights[inputNode][hiddenNode];
      }

      // Apply activation function and store in output data for hidden layer 0.
      this.hiddenLayers[0].outputData[hiddenNode] = this.hiddenLayers[0].activationFunction(
        this.hiddenLayers[0].inputData[hiddenNode]);
      console.log(this.hiddenLayers[0].outputData[hiddenNode], ' ', this.hiddenLayers[0].inputData[hiddenNode]);
    }

    // Loop over each [1..n] hidden layer.
    for (let hiddenLayer = 1; hiddenLayer < this.hiddenLayers.length; hiddenLayer++) {
      this.hiddenLayers[hiddenLayer].clearInputAndOutputData();
      for (let hiddenNode = 0; hiddenNode < this.hiddenLayers[hiddenLayer].count; hiddenNode++) {
        // Add bias to the input for this node in hidden layer <hiddenLayer>
        if (this.hiddenLayers[hiddenLayer].biasWeights !== undefined) {
          this.hiddenLayers[hiddenLayer].inputData[hiddenNode] += this.hiddenLayers[hiddenLayer - 1].biasWeights[hiddenNode];
        }
        // Loop over each input node in the previous hidden layer.
        for (let prevHiddenNode = 0; prevHiddenNode < this.hiddenLayers[hiddenLayer - 1].count; prevHiddenNode++) {
          this.hiddenLayers[hiddenLayer].inputData[hiddenNode] += this.hiddenLayers[hiddenLayer - 1].outputData[prevHiddenNode]
            * this.hiddenLayers[hiddenLayer - 1].weights[prevHiddenNode][hiddenNode];
        }
        // Apply activation function and store in output data for hidden layer <hiddenLayer>.
        this.hiddenLayers[hiddenLayer].outputData = this.hiddenLayers[hiddenLayer]
          .inputData.map(this.hiddenLayers[hiddenLayer].activationFunction);
      }
    }

    this.outputLayer.inputData = [];
    this.outputLayer.outputData = [];

    let lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1]
    // Loop over each output node.
    for (let outputNode = 0; outputNode < this.hiddenLayers[0].count; outputNode++) {
      // Add bias to the input for this node in hidden layer 0.
      if (lastHiddenLayer.biasWeights !== undefined) {
        this.outputLayer.inputData[outputNode] += lastHiddenLayer.biasWeights[outputNode];
      }

      // Loop over each input node in the input layer.
      for (let hiddenNode = 0; hiddenNode < lastHiddenLayer.count; hiddenNode++) {
        this.outputLayer.inputData[outputNode] += lastHiddenLayer.weights[hiddenNode][outputNode];
      }
      // Apply activation function.
      this.outputLayer.outputData = this.outputLayer.inputData.map(this.outputLayer.activationFunction);
    }

    let error = data.expectedOutputs.map((eOut, i) => this.errorCalculator(this.outputLayer.outputData[i], eOut));
    console.log('error: ', error);
    return { result: this.outputLayer.outputData, error };
  }

  /*
  const relu: (input: number) => number = (n => Math.max(0, n));

  const input: number = 1;
  const inputLayerWeight: number = 0.5;

  const hiddenLayerInput = input * inputLayerWeight;
  const hiddenLayerOutput = relu(hiddenLayerInput);
  const hiddenLayerWeight: number = 0.5;

  const outputLayerInput = hiddenLayerOutput * hiddenLayerWeight;
  const outputLayerOutput = relu(outputLayerInput);

  const expectedValue = 1.0;
  const error = expectedValue - outputLayerOutput;

  console.log(error);
*/
  public train(data: {inputs: number[], expectedOutputs: number[]}): {result: number[], error: number[]} {
    /*
      let dError1WRTOutput1 = +/-1 = dError(output)

      let dOutput1WRTOutput1Input = outputLayerInput > 0 ? 1 : 0 [dActivationFunction(outputLayerInput)]

      let dOutput1InputWRThidden1ToOutput1Weight = hiddenLayer1Output [previousLayerOutput]

      let dHiddenLayer1OutputWRTHiddenLayer1Input = hiddenLayer1Input > 0 ? 1 : 0 [dActivationfunction(previousLayerOutput)]

      let dHiddenLayer1InputWRTInputLayerWeight = input [previousLayerOutput]
    */

      let result = this.feedForward(data);
      let error = result.error;

      // TODO: Implement error function deltas...
      // TODO: Use last weight in each layer's node list to set the weight of the bias nodes in each layer...

      let outputDeltas = this.outputLayer.outputData.map((_, i) => {
        return (error[i] > 0 ? 1 : -1) * this.outputLayer.deltaActivationFunction(this.outputLayer.inputData[i]);
      });

      // weightDeltas: number[hiddenLayerNode][outputNode];

      for (let hiddenLayer = this.hiddenLayers.length - 1; hiddenLayer > 0; hiddenLayer--) {
        // Use output layer deltas.
        if (hiddenLayer === this.hiddenLayers.length - 1) {
          this.hiddenLayers[hiddenLayer].weightDeltas = this.hiddenLayers[hiddenLayer].weightDeltas.map((arr, node) => {
            return arr.map((_, outputNode) => this.hiddenLayers[hiddenLayer].outputData[node] * outputDeltas[outputNode]);
          });
        } else { // Use next hidden layer deltas.
          let nextHiddenLayer = this.hiddenLayers[hiddenLayer + 1];
          this.hiddenLayers[hiddenLayer].weightDeltas = this.hiddenLayers[hiddenLayer].weightDeltas.map((arr, node) => {
            return arr.map((_, nextLayerNode) => {
              // Find sum of weight deltas for the next layer's node
              let dNextLayerNodeWeights = nextHiddenLayer.weightDeltas[nextLayerNode].reduce(MathHelper.add);
              let dNextLayerNodeOutput = nextHiddenLayer.deltaActivationFunction(nextHiddenLayer.inputData[nextLayerNode]);
              return this.hiddenLayers[hiddenLayer].outputData[nextLayerNode] * dNextLayerNodeWeights * dNextLayerNodeOutput;
            });
          });
        }
      }

      // Calculate weight deltas for the input layer.
      // weightDeltas: number[inputLayerNode][hiddenLayerNode];
      let firstHiddenLayer = this.hiddenLayers[0];
      this.inputLayer.weightDeltas = firstHiddenLayer.weightDeltas.map((arr, node) => {
        return arr.map((_, nextLayerNode) => {
          // Find sum of weight deltas for the next layer's node
          let dNextLayerNodeWeights = firstHiddenLayer.weightDeltas[nextLayerNode].reduce(MathHelper.add);
          let dNextLayerNodeOutput = firstHiddenLayer.deltaActivationFunction(firstHiddenLayer.inputData[nextLayerNode]);
          return this.inputLayer.inputData[node] * dNextLayerNodeWeights * dNextLayerNodeOutput;
        });
      });

      return { result: this.outputLayer.outputData, error };

  }
}