import {defineConfig} from "vite";
import {cloudflare} from "@cloudflare/vite-plugin";

export default defineConfig({
  root: "web",
  plugins: [cloudflare()],
  publicDir: "public",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 8090,
    strictPort: true,
  },
});
