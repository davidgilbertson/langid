import {defineConfig} from "vite";
import {resolve} from "node:path";
import {cloudflare} from "@cloudflare/vite-plugin";

export default defineConfig({
  root: "web",
  plugins: [cloudflare()],
  publicDir: "public",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve("web/index.html"),
        predictLanguage: resolve("web/predictLanguage.js"),
        modelViz: resolve("web/modelViz.js"),
      },
    },
  },
  server: {
    port: 8090,
    strictPort: true,
  },
});
