# Deploying to a free URL you can open on any device

The pipeline is batch work that produces one HTML file, so it does not need a
server. GitHub Actions runs it on a schedule and GitHub Pages serves the
result. Total cost: nothing.

Your five weekly slate runs use roughly 100 of the 2,000 free Actions minutes
per month.

## One-time setup (about 10 minutes)

**1. Create the repo**

```bash
cd ~/nflproj
git init
git add .
git commit -m "nflproj"
```

Create an empty repo on github.com, then:

```bash
git remote add origin https://github.com/<you>/nflproj.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes the large parquet artifacts and cached data, so
only code ships — about 700KB.

**2. Turn on Pages**

Repo → Settings → Pages → Source: **GitHub Actions**.

**3. Run it**

Actions tab → "NFL projections" → **Run workflow**.

Your dashboard is then live at:

```
https://<you>.github.io/nflproj/
```

Open it on any device. On iOS, Share → Add to Home Screen gives it an app icon.

## What runs automatically

| Slate | Cron (UTC) | ET |
|---|---|---|
| TNF | `15 21 * * 4` | Thu 5:15pm |
| Sunday early | `0 15 * * 0` | Sun 11:00am |
| Sunday late | `5 18 * * 0` | Sun 2:05pm |
| SNF | `20 21 * * 0` | Sun 5:20pm |
| MNF | `15 21 * * 1` | Mon 5:15pm |

Each run refreshes nflverse data, runs the leakage audit, projects the next
slate that has not kicked off, and publishes. **The audit gates the deploy** —
if any of the nine tests fail, nothing is published and the run shows red.

You can also trigger a run by hand from the Actions tab, including from your
phone's browser.

## Two caveats worth knowing

**The Refresh button will not work on Pages.** Pages serves static files and
cannot execute Python. The scheduled runs replace it. If you want the button,
run `python3 -m nflproj.cli serve` locally, or use the Spaces option below.

**Cron in Actions is best-effort**, and can be delayed by several minutes when
GitHub is busy. The Sunday 11:00am run is the one that matters most
(inactives), so consider scheduling it at 10:45 to be safe.

**Daylight saving:** the crons above are UTC and assume EDT. After the November
change, add one hour to each UTC time.

## If you want the repo private

GitHub Pages on a private repo needs a paid plan. Two free alternatives:

- **Cloudflare Pages** — free, works with private repos. Replace the deploy job
  with `cloudflare/wrangler-action` uploading the `public/` directory.
- **Netlify** — free tier, same approach.

Both keep the Actions workflow exactly as written; only the final deploy step
changes.

## If you want the interactive version

**Hugging Face Spaces** is free and can run Python, so the Refresh button
works. Create a Space (SDK: Docker or Gradio), push the same code, and run
`nflproj serve` bound to `0.0.0.0:7860`. Free Spaces sleep after inactivity and
wake on first request, which adds a few seconds to the first load.

This is the only option here that gives you a live button on any device for
free. GitHub Pages is more reliable for scheduled output; Spaces is better for
on-demand runs.
