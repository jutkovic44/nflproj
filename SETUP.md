# Setup — from download to a live URL

Everything below is free. Two paths: **A** (public repo, simplest) or
**B** (private repo, one extra service).

---

## Step 0 — unpack

```bash
cd ~/Downloads
unzip -o nflproj-deploy.zip -d ~
cd ~/nflproj-deploy
```

You should see: `nflproj/`  `.github/`  `requirements.txt`  `DEPLOY.md`

---

## Step 1 — install and verify locally (2 minutes)

```bash
pip3 install -r requirements.txt
python3 -m nflproj.cli refresh
python3 -m nflproj.cli audit
```

Expect **nine PASS lines and `audit passed`**. If the audit fails, stop —
do not deploy. Send me the output.

```bash
python3 -m nflproj.cli project --next --dashboard public/index.html
open public/index.html
```

That is exactly what the server will run.

---

## Step 2a — PUBLIC repo (recommended)

With the GitHub CLI (`brew install gh` then `gh auth login` once):

```bash
cd ~/nflproj-deploy
git init && git add . && git commit -m "nflproj"
gh repo create nflproj --public --source=. --push
```

Without the CLI: create an empty repo named `nflproj` on github.com, then:

```bash
cd ~/nflproj-deploy
git init && git add . && git commit -m "nflproj"
git remote add origin https://github.com/<YOU>/nflproj.git
git branch -M main && git push -u origin main
```

Then in the browser:

1. Repo → **Settings** → **Pages**
2. Source: **GitHub Actions**
3. Repo → **Actions** → "NFL projections" → **Run workflow**

Live at `https://<YOU>.github.io/nflproj/` in about three minutes.

---

## Step 2b — PRIVATE repo

Same push, with `--private`. GitHub Pages needs a paid plan for private repos,
so deploy to Cloudflare Pages instead (free, private-repo friendly):

1. Create a free Cloudflare account → Workers & Pages → Create → Pages
2. Direct Upload, name the project `nflproj`
3. Get an API token (My Profile → API Tokens → Edit Cloudflare Workers)
4. In GitHub: Settings → Secrets and variables → Actions → add
   `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`
5. In `.github/workflows/slate.yml`, replace the whole `deploy:` job with:

```yaml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: github-pages, path: public }
      - uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          command: pages deploy public --project-name=nflproj
```

Live at `https://nflproj.pages.dev`.

---

## Step 3 — on your phone

Open the URL, then Share → **Add to Home Screen**. It behaves like an app.

---

## What happens automatically

| Slate | ET |
|---|---|
| TNF | Thu 5:15pm |
| Sunday early | Sun 11:00am |
| Sunday late | Sun 2:05pm |
| SNF | Sun 5:20pm |
| MNF | Mon 5:15pm |

Each run refreshes nflverse data, runs the nine-test leakage audit, projects
the next slate that has not kicked off, and publishes. **The audit gates the
deploy** — a failure publishes nothing and shows the run red.

Manual run any time: Actions tab → Run workflow. Works from your phone's
browser.

---

## Things that will bite you

**Refresh button does nothing on the hosted page.** Static hosting cannot run
Python. Use the schedule, or run `python3 -m nflproj.cli serve` locally when
you want the button.

**Crons are UTC and assume EDT.** After the November time change, add one hour
to each cron in `slate.yml`.

**Actions cron is best-effort** and can run minutes late. The Sunday 11:00am
run is the important one (inactives are final 90 minutes out); move it to
10:45 if you want margin.

**The pipeline runs V3.1**, the production point model. The experimental V4.x
distributional work is in the package but is not wired to the dashboard, which
is deliberate.
