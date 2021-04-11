
export class ArrayHelper {
  static avg(arr: number[]) {
    let sum = 0;
    arr.forEach(n => sum += n);
    return sum / arr.length;
  }

  static init<T>(n: number, f: (index: number) => T) {
    const arr: T[] = [];
    for (let i = 0; i < n; i++) {
      arr.push(f(i));
    }
    return arr;
  }

  static init2D<T>(m: number, n: number, f: (i: number, j: number) => T) {
    return ArrayHelper.init(m, i => ArrayHelper.init(n, j => f(i, j)));
  }

  static min(arr: number[]): number {
    let min = arr[0];
    for (let n of arr) {
      if (n < min) {
        min = n;
      }
    }
    return min;
  }
}

export class MathHelper {
  /**
   * Return a random integer in range [a, b)
   * @param a Inclusive minimum value of the random number
   * @param b Exclusive maximum value of the random number
   */
  static randInt(a: number, b: number): number {
    return Math.floor(Math.random() * b) + a;
  }

  static add(a: number, b: number): number {
    return a + b;
  }
}
