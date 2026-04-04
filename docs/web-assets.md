# Web asset workflow

Cyber-Inference vendors the required UI assets into the repo so the web UI does not depend on runtime third-party CSS/JS/font loading.

## Repo-owned assets
- Source stylesheet: `src/cyber_inference/web/assets/app.css`
- Generated stylesheet: `src/cyber_inference/web/static/css/app.css`
- Local fonts: `src/cyber_inference/web/static/fonts/`
- Local images: `src/cyber_inference/web/static/images/`
- Tailwind theme config: `tailwind.config.js`

## Rebuild command
```bash
./scripts/build_web_assets.sh
```

This uses `npx tailwindcss@3.4.17` at build time to regenerate the local CSS bundle. The runtime UI only loads repo-owned static assets.
