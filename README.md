# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cahya23/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cahya23/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 November 2026, UTC), as five quantiles (5%, 25%, 50%, 75%, 95%) of the concurrent-viewer count for hours 2 to 24.

## Method

The program reads one case file (`/input/<id>.in`) and writes one forecast (`/output/<id>.out`). It uses only what the case gives it: hour 1's print as the anchor, and the 42 past big launches in `comparables` as the library of launch-day shapes.

For each past launch it computes the log-ratio of every hour's viewers to hour 1's. The forecast for an hour is the weighted quantiles of those ratios, anchored on hour 1 — weighted by how close each past launch's hour-1 audience is to this launch's (a Gaussian kernel in log space, bandwidth 1.0), because a launch already huge at midnight UTC grows differently from one that is nearly empty then. Two guards keep it honest: the bandwidth widens if the weights concentrate on fewer than 5 games, and the log-quantiles are smoothed over neighbouring hours. With no usable comparables it falls back to a flat persistence forecast. Nothing is random; two runs write the same bytes.

Validated offline on public SullyGnome history (the same source the cases use), by running this exact program on the four practice cases: mean pinball loss 0.232 on the four practice launches, against 0.387 for the house baseline; leave-one-out over the 37 comparables with a usable release day gives 0.259 against 0.321 for the baseline. The practice answers are public history; the program itself only ever reads the case file it is given. With no usable comparables in a case it falls back to a flat persistence forecast, so it never crashes and never writes an invalid case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the input directory and writes one `<id>.out` per case.

## Sources

- Twitch category counts: SullyGnome.com hourly Twitch category averages (public directory figures, quoted with attribution).
- Launch facts: GTA VI releases 19 November 2026 on PlayStation 5 and Xbox Series X|S, rolling midnight-local unlock; pre-orders at unprecedented levels (Take-Two, November 2026).

---

<sub>Entered on [make.fun](https://make.fun/?ref=b7cc1mhs&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
