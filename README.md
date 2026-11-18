# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calistaa113/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calistaa113/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first hour. `main.py`
reads the case file (`/input/<id>.in`) and writes five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) of each hour 2–24 to `/output/<id>.out`.

## Method

The launch-day shape is learned from the case's own comparables — past big
launches' hourly Twitch counts. For each comparable with a full release
day, take the log ratio of each hour's viewers to its hour-1 viewers. That
shape depends strongly on how big hour 1 already is (correlation of
log hour-1 with the log ratio is about −0.8 to −0.9 by the evening): a
launch already huge at 00:00 UTC grows little, a small one grows
several-fold as the US unlocks and prime time arrives. So each hour's
log ratio is predicted by a local linear regression on log hour-1 over the
18 comparables nearest in log hour-1 size, with the log hour-1 capped at
the largest neighbour (no extrapolation past the data), and the spread from
the local residuals (1.4826 × MAD, floored at 0.10 log units). No network,
no model calls, no clock, no randomness — the same case always gives the
same bytes.

Leave-one-out backtest over the kit's 37 usable comparables (each held out
in turn, trained on the other 36): mean pinball loss **0.382** vs **0.930**
for the persistence baseline, typical median miss 73% vs 213%. The practice
cases' own answers are not in the kit (the server grades them), so this is
the honest local check available.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. One run over the kit's four practice
cases takes about 0.25 s; the grader's cap is 8 s per case.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
