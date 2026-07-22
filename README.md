# ERCOT Market Intelligence and Battery Arbitrage Platform

Forecasting day ahead power prices and optimising grid battery arbitrage on real ERCOT market data.

This project takes wholesale electricity prices from ERCOT, the grid operator for most of Texas.
It forecasts the next day of prices, then works out how a grid battery should charge and discharge
to earn the most from the price swings. It runs end to end, from downloading raw market data, to a
tested price forecast, an exact battery optimiser, a backtest, a small web service, and a dashboard
you can click through.

**[Open the live dashboard](https://ercot-battery-arbitrage.streamlit.app)**, or run it locally with `make dashboard`.

## The idea

A battery only earns when prices move, and it can only act on the prices it can predict. So the
number that matters most is how much of the best possible profit a real operator can actually reach.

The backtest reports two numbers. The first is a perfect foresight ceiling, the most a battery could
earn if it knew every price in advance. The second is what a forecast driven operator earns, choosing
its schedule from a forecast and then living with the real prices. The gap between them is the cost of
not seeing the future. The ratio between them is the capture rate, which is how the industry grades a
battery operator. On this data the operator captures about 70 percent of the ceiling across the year.
A capture rate near 100 percent is not a good sign, it usually means future prices have leaked into
the test.

## Why ERCOT, and why I built it

ERCOT is the most volatile large power market in the United States, and it has no capacity market. So
a battery there earns almost all of its money from how far and how often prices swing inside a day. In
this project's two years of data the average hub price is about 30 dollars per MWh. Yet day ahead
prices spike past 2,000 dollars on the worst days, and real time prices past 4,900, and the ten best
days alone carry about a sixth of the whole year of battery profit. Low average prices broken by a few
big days is exactly the setting where a careless model does badly and careful work pays off.

I built it to answer the capture rate question end to end, on real data. Batteries are the fastest
growing asset on the grid and their business case is arbitrage, so the useful number is not the
theoretical maximum but how close a real operator gets to it. It also works as a portfolio piece. It
shows data engineering across messy public sources, price forecasting judged against a published
benchmark, and a battery optimisation and backtest built the way the industry frames the problem.

## What it looks like

The dashboard has four tabs, all live at the link above. You pick a forecast model once in the
sidebar, and the tabs follow that choice. Everything below is drawn from the real data, the ERCOT
hub average across 2024 and 2025.

**Prices.** The day ahead and real time price history for the chosen hub, with the latest price of
each shown above its chart. Prices sit low and calm for long stretches, near 30 dollars per MWh, then
spike hard on the worst days, past 2,000 dollars in the day ahead market and past 4,900 in real time.
That pattern, quiet prices broken by sharp peaks, is the whole reason a battery earns anything.

**Forecast.** The forecast against what really happened for the chosen model, with the error scores
above. The default model is off by about 12 dollars per MWh on average. Its error measured against a
naive same hour last week guess is 0.85, so it makes about fifteen percent less error than that baseline. The
line chart shows the predicted and the real prices for one delivery day tracking each other closely
through the overnight lull and up into the evening peak.

**Dispatch.** The battery schedule for a chosen day. The top chart is the real price through the day,
the middle chart is the charge and discharge power, and the bottom chart is the state of charge
filling and emptying. The battery buys in the cheap hours and sells into the evening peak. The tab
opens on a typical day, the one with the median profit, where the forecast picked the same hours as
perfect foresight and so captured 100 percent. Across the whole year the capture rate is about 70
percent, which is the headline.

**Results.** The headline numbers. The table is annualised profit per kW of battery, swept across one,
two, and four hour batteries, with and without a cycling cost, for both the ceiling and the forecast
driven operator. A two hour battery earns about 40 dollars per kW year at the ceiling, and about 29 as
a forecast driven operator. Longer batteries earn more, and a cycling cost trims the total sharply,
far enough to wipe out the forecast driven profit. The two curves below plot cumulative profit against
days ranked from best to worst, and both show the known result that a small share of days carry most
of the year.

## How it works

**Collecting the data.** Downloads the raw inputs. ERCOT day ahead and real time prices, the day ahead
demand forecast, and weather zone temperature all come from the hosted gridstatus.io API, which serves
years of history the free public feed cannot reach. Every raw file is saved by date and never
overwritten, so the history stays fixed and any run can be repeated later.

**Cleaning it.** Turns the raw files into tidy price tables. It parses every timestamp to one time
zone, removes the duplicated clock hour that ERCOT publishes at the daylight saving change, checks the
columns, types, and value ranges, and tags each row with the market rule in force at the time. Negative
prices are kept, because they are real in ERCOT.

**Building the model inputs.** Builds the table the forecast learns from. This is recent prices at set
delays, calendar facts like the hour and the day of the week, and the demand forecast and temperature
for the day being predicted. A guard checks that no piece of information from the future can reach a
row, so training cannot cheat.

**Forecasting tomorrow's prices.** Predicts the next day of prices. It trains a gradient boosted tree
model and a published linear model that is a standard reference in the field, and it measures both
against two simple baselines. The test walks forward through time, so every day is predicted using
only the data that existed before it, never a random split that would leak later prices backwards.

**Scheduling the battery.** Given a day of prices and a battery, this finds the charge and discharge
plan that earns the most. It is written as an exact optimisation, not a rule of thumb, so the answer is
the true best plan for those prices. One rule stops the battery charging and discharging at the same
time. Another makes the battery end each day at the same charge it started, so no day can look better
by selling off a full battery it began with.

**Running the backtest.** Runs the optimiser across the history in the two ways that matter. The
ceiling plan is solved on the real prices, which is the best anyone could have done while knowing the
future. The forecast driven plan is solved on the forecast, then settled against the real prices, which
is what a real operator would have earned. The distance between them is the cost of forecast error, and
their ratio is the capture rate. One check refuses to run across two different market rule periods at
once. Another refuses to accept a forecast driven profit above the ceiling.

**Serving the results.** A small read only web service returns the finished tables as data. It never
calculates anything inside a request, it just serves what the pipeline already produced, so it stays
fast and cannot drift from the stored results.

**The dashboard.** A web page that lets anyone explore the prices, the forecast against reality, a day
of battery dispatch, and the headline results without touching the code.

## Assumptions

**The battery.** The default is a 1 MW battery that holds two hours of energy, with an 85 percent round
trip efficiency split evenly across charging and discharging. It starts and ends each day at the same
state of charge. Results are reported per unit of power so they scale to any size, and the results view
also runs a one hour and a four hour battery for comparison.

**The market and the horizon.** The arbitrage backtest uses day ahead hourly prices. Day ahead clears
once a day, which keeps the framing simple. Forecast the next day, schedule against the forecast, then
settle against what actually cleared. Real time prices are collected and shown, but the headline
backtest is on day ahead.

**The location.** The price is an ERCOT hub average rather than a single node, so the result
reflects system wide value and not one location's congestion quirks.

**Market rule changes.** ERCOT has changed its price cap over the years, and each period has a
different price distribution. The default backtest window sits entirely inside one period, and the code
refuses to blend two periods into a single result without saying so.

**Limits.** The battery is treated as a price taker that does not move the market, which is fair for a
small asset. The ceiling assumes perfect foresight and perfect dispatch, which is why it is a ceiling
and not a plan. Efficiency is a single constant. Battery ageing is reported rather than charged against
the headline profit. Ancillary service income is left out. Each day is scheduled on its own, with no
carrying of energy across midnight. The two daylight saving change days each year, which run twenty
three and twenty five hours rather than a clean twenty four, are left out of the backtest. Each of
these is stated plainly so the numbers are read for what they are.

## Setup

```
make setup
```

This creates a local environment and installs the pinned dependencies. Then copy `.env.example` to
`.env` and add a gridstatus.io API key, which the data collection step needs.

## Seeing it run

The finished tables from a full real run are already in the repository, the processed prices and the
results. So you can see everything straight away. Open the live dashboard linked at the top of this
page, or run it locally with the command in the dashboard section below. Nothing has to be rebuilt
first.

## Running the real pipeline

Each stage has its own command. They run in order, and each one writes the tables the next one reads.
Rebuilding from raw needs the gridstatus.io API key from the setup step.

```
make ingest       # download the raw market data
make build        # clean it into price tables
make features     # build the model input table
make forecast     # train the models and forecast prices
make backtest     # run the arbitrage backtest
```

## The dashboard

```
make dashboard
```

This opens the dashboard in your browser at http://localhost:8501. Pick a forecast model in the
sidebar, then move through the four tabs described above. A hosted version can be deployed to
Streamlit Community Cloud, and once it is the live link sits at the top of this page.

## The API

```
make serve
```

This starts the read only service at http://localhost:8000. It answers on a health check, the latest
prices for a hub, the latest forecast against real prices, and the headline backtest summary. Times are
given in UTC in ISO 8601 form, missing numbers come back as null, and a table the pipeline has not
produced yet returns a clean not found rather than an error.

## Testing

```
make test              # the whole suite
make test-forecast     # one part on its own
```

Every part of the pipeline has its own test group, so any part can be checked on its own. The suite has
over a hundred tests across nine groups, config, ingest, validate, features, forecast, optimise,
backtest, api, and dashboard.

## Configuration

Every setting lives in a set of small config files and is loaded into typed, checked objects. So a
single mistyped setting fails at once with a clear message, rather than deep inside a model run. Nothing
is hard coded, so the battery size, the price hubs, the model settings, and the backtest window can all
be changed in one place.

## Data sources

- ERCOT day ahead and real time prices, the day ahead demand forecast, and weather zone temperature,
  all from the hosted gridstatus.io API.

## Refreshing the data

`make refresh` runs the whole pipeline end to end from the local machine. It downloads the latest
data, rebuilds every table, and writes the refreshed results in place. It needs the gridstatus.io key
from the setup step, and the raw history the pipeline builds on stays on the machine that runs it.

## Continuous integration

Every push and pull request runs the lint and the full test suite on GitHub Actions. The suite uses
synthetic and fixture data, so it needs no API keys and no network, which makes it a fast and
reproducible check that the whole pipeline still works.
