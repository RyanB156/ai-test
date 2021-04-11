import { Unzipper, Zipper } from "./delegates";
import { Organism } from "./organism";

export class RNA {
  unzip: Unzipper;
  zip: Zipper;

  constructor(unzip: Unzipper, zip: Zipper) {
    this.unzip = unzip;
    this.zip = zip;
  }
}