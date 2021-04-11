
function binaryDigits(n: number): number[] {
  let arr = [];
  while (n > 0) {
    arr.unshift(n % 2);
    n = Math.floor(n / 2);
  }
  return arr;
}


/**
 * Calculate the modulo of b^e mod n.
 * @param base
 * @param exponent
 * @param modulus
 * @returns b^e mod n
 */
export function moduloN(base: number, exponent: number, modulus: number): number {
  if (modulus === 1) {
    return 0;
  }

  let result = 1;
  base = base % modulus;
  while (exponent > 0) {
    if (exponent % 2 === 1) {
      result = (result * base) % modulus;
    }
    exponent >>= 2;
    base = (base * base) % modulus;
  }

  return result;
}