import test from "node:test";
import assert from "node:assert/strict";
import { VERSION } from "../src/index.ts";

test("package exports a version", () => {
  assert.equal(VERSION, "0.0.1");
});
