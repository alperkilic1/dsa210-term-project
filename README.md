# DSA210 Term Project — Incident Genome

## What is this?
I want to look at public cloud service outages (like GitHub going down, or OpenAI having issues) and try to find patterns. When a service goes down, can we predict how long it will take to fix based on what we know early on?

## Why this topic?
I use GitHub, OpenAI and Cloudflare almost every day. When they go down I always wonder — is this going to be a 10-minute fix or a 5-hour disaster? I thought it would be cool to actually look at the data and find out.

## Where does the data come from?
Many cloud services have public status pages that give you incident history as JSON. No API key needed, just a URL. For example:
- `https://www.githubstatus.com/api/v2/incidents.json`
- `https://status.openai.com/api/v2/incidents.json`
- `https://www.cloudflarestatus.com/api/v2/incidents.json`

I plan to collect data from around 15 services.

## What I plan to do
- Collect incident data and clean it up
- Look at patterns: which components fail most? Do outages happen more at certain times?
- Test some hypotheses (e.g. do incidents with more affected components last longer?)
- Try to build a simple model that predicts if an outage will be short or long

## Tools
Python, pandas, matplotlib, scikit-learn

## AI Assistance
I used ChatGPT and Claude for help with debugging the data collection script and formatting the proposal document. All project ideas, data source selection, and analysis planning were done by me.
