import { ArrayHelper } from "../helpers";
import { Matrix } from "../linear-algebra/matrix";

export function tsp(matrix: Matrix): number[] {


  return [];
}


export function minDistOfGraph(start: number, weights: Matrix) {

  let nodeCount = weights.data[0].length;

  function getNeighbours(node: number, options: number[]) {
    let nodeIndex = options.findIndex(n => n === node);
    return options.slice(0, nodeIndex).concat(options.slice(nodeIndex + 1));
  }

  function minDist(start: number, neighbors: number[]): number {
    return ArrayHelper.min(neighbors.map(n => weights.data[start][n] + minDist(n, getNeighbours(n, neighbors))));
  }

  let nodes = ArrayHelper.init(nodeCount, i => i);
  return minDist(start, getNeighbours(start, nodes));

}