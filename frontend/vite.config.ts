import { defineConfig } from "vite";

// base "./" + relativní cesty k API → aplikace funguje i pod prefixem (např. /l16/).
export default defineConfig({
  base: "./",
  server: {
    proxy: { "/api": process.env.L16_API ?? "http://localhost:8000" },
  },
});
