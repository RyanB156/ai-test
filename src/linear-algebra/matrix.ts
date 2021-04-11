import { ArrayHelper } from '../helpers';

export class Matrix {
  m: number; // Number of rows.
  n: number; // Number of columns.
  data: number[][]; // Data [row][col].

  constructor(m: number, n: number, data: number[][]) {
    this.m = m;
    this.n = n;
    this.data = data;
  }

  static empty(m: number, n: number): Matrix {
    return new Matrix(m, n, ArrayHelper.init2D(m, n, _ => 0));
  }

  toString(): string {
    let str = `{m: ${this.m}, n: ${this.n}\n`;
    for (let i = 0; i < this.m; i++) {
      str += '['
      for (let j = 0; j < this.n; j++) {
        str += this.data[i][j].toString() + (j < this.n - 1 ? ', ': '');
      }
      str += ']' + (i < this.m - 1 ? ',\n': '');
    }
    return str + '}';

  }

  add(matrix: Matrix): Matrix {
    if (this.m !== matrix.m || this.n !== matrix.n) {
      throw Error('Matrix dimensions did not match');
    }

    let result: Matrix = Matrix.empty(this.m, this.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.data[i][j] = this.data[i][j] + matrix.data[i][j];
      }
    }
    return result;
  }

  subtract(matrix: Matrix): Matrix {
    if (this.m !== matrix.m || this.n !== matrix.n) {
      throw Error('Matrix dimensions did not match');
    }

    let result: Matrix = Matrix.empty(this.m, this.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.data[i][j] = this.data[i][j] - matrix.data[i][j];
      }
    }
    return result;
  }

  multiply(matrix: Matrix): Matrix {

    // console.log(`${this.toString()} * ${matrix.toString()}`);

    if (this.n !== matrix.m) {
      throw Error(`Matrix dimensions did not match\nthis: ${this.toString()}\nmatrix: ${matrix.toString()}\n`);
    }

    let result: Matrix = Matrix.empty(this.m, matrix.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < matrix.n; j++) {
        for (let k = 0; k < this.n; k++) {
          result.data[i][j] += this.data[i][k] * matrix.data[k][j];
        }
      }
    }
    return result;
  }

  scale(scale: number): Matrix {
    let result: Matrix = Matrix.empty(this.m, this.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.data[i][j] = this.data[i][j] * scale;
      }
    }
    return result;
  }

  transpose(): Matrix {
    let result: Matrix = Matrix.empty(this.n, this.m);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.data[j][i] = this.data[i][j];
      }
    }
    return result;
  }

  map(f: (i: number) => number): Matrix {
    let result: Matrix = Matrix.empty(this.m, this.n);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result.data[i][j] = f(this.data[i][j]);
      }
    }
    return result;
  }
}