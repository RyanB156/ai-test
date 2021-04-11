/**
 * Specifies the input layer structure of a network.
 */
export class InputStructure {
  count: number;
  biasEnabled: boolean;

  constructor(count: number, biasEnabled: boolean) {
    this.count = count;
    this.biasEnabled = biasEnabled;
  }
}

/**
 * Specifies the hidden layer structures of a network.
 */
export class HiddenStructure {
  count: number;
  biasEnabled: boolean;
  activationFunctionName: string;

  constructor(count: number, biasEnabled: boolean, activationFunctionName: string) {
    this.count = count;
    this.biasEnabled = biasEnabled;
    this.activationFunctionName = activationFunctionName;
  }
}

/**
 * Specifies the output layer structure of a network.
 */
export class OutputStructure {
  count: number;
  activationFunctionName: string;

  constructor(count: number, activationFunctionName: string) {
    this.count = count;
    this.activationFunctionName = activationFunctionName;
  }
}

/**
 * Specifies the structure of a network.
 */
export class NeuralNetworkStructure {
  inputLayer: InputStructure;
  hiddenLayers: HiddenStructure[];
  outputLayer: OutputStructure;
  errorFunctionName: string;

  constructor(inputLayer: InputStructure, hiddenLayers: HiddenStructure[],
    outputLayer: OutputStructure, errorFunctionName: string) {
      this.inputLayer = inputLayer;
      this.hiddenLayers = hiddenLayers;
      this.outputLayer = outputLayer;
      this.errorFunctionName = errorFunctionName;
  }
}