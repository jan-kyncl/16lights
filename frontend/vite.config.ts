import { readFileSync } from "node:fs";
import { defineConfig } from "vite";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")) as {
  version: string;
};

// base "./" + relativní cesty k API → aplikace funguje i pod prefixem (např. /l16/).
export default defineConfig({
  base: "./",
  define: { __APP_VERSION__: JSON.stringify(pkg.version) },
  server: {
    proxy: { "/api": process.env.L16_API ?? "http://localhost:8000" },
  },
});
