# DSA210 Term Project Proposal

**Alper Kılıç**

## Where will I get the data?

I'm going to use public status page APIs from cloud services like GitHub, OpenAI, Cloudflare, Discord, etc. These services have a public endpoint (`/api/v2/incidents.json`) that gives you their incident history as JSON without needing any API key. I'm planning to collect from about 10-15 services.

## How will I collect it?

I wrote a Python script that goes through each service's API and pulls all the incident records. It paginates through the results and saves the raw JSON files. Then I parse each incident to get things like when it started, when it was resolved, how many components were affected, and the severity level.

## Data characteristics

So far I've done a quick test run and got 359 incidents from 8 services. The full collection should give around 1500-3000 incidents going back a few years. Each incident has about 15 features — stuff like duration (in minutes), impact level (minor/major/critical), number of status updates, day of week, hour of day, etc. Some services have way longer average outages than others (GitHub averages about 2 hours, but Atlassian averages over 20 hours which is crazy). I want to see if I can predict whether an outage will be short or long based on the early signals.
