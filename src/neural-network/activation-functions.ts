export class ActivationFunctions {

  static names = {
    relu: 'relu',
    identity: 'identity'
  }

  static relu(n: number): number {
    return Math.max(0, n);
  }
  static reluDelta(n: number) {
    return n > 0 ? 1 : 0;
  }

  static identity(n: number) {
    return n;
  }
  static identityDelta(n: number) {
    return 1;
  }

  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
  static sigmoidDelta(x: number): number {
    return this.sigmoid(x) * (1 - this.sigmoid(x));
  }

  static map(name: string): [((n: number) => number), ((n: number) => number)] {
    switch (name) {
      case ActivationFunctions.names.relu:
        return [ActivationFunctions.relu, ActivationFunctions.reluDelta];
      default:
        return [ActivationFunctions.identity, ActivationFunctions.identityDelta];
    }
  }
}