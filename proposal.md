# DSA210 Term Project Proposal — Incident Genome

**Student:** Alper Kılıç
**Date:** March 28, 2026

## Data Source

The data comes from public Statuspage API endpoints that major cloud services use to report incidents. These APIs are freely accessible without authentication and return structured JSON data containing incident details, affected components, timeline updates, and resolution status. I will collect data from 15 services: GitHub, OpenAI, Cloudflare, Twilio, Datadog, Atlassian (Jira/Confluence), Reddit, Discord, Dropbox, Vercel, Netlify, Heroku, DigitalOcean, Linear, and Notion.

## Data Collection Method

A Python script (`collect_data.py`) sends paginated GET requests to each service's `/api/v2/incidents.json` endpoint, collecting all available historical incidents. Each incident record includes: creation time, resolution time, impact level (none/minor/major/critical), number of status updates, affected components, and the first update text. The script computes derived features such as incident duration (in minutes), hour/weekday of occurrence, and component count. Raw JSON responses are saved for reproducibility, and parsed records are stored in a unified `incidents.json` file.

## Data Characteristics

Based on preliminary collection from 8 accessible services (GitHub, OpenAI, Cloudflare, Discord, Atlassian, Reddit, Vercel, Netlify), the first page alone yields 359 incidents. Full pagination is expected to produce 1500–3000 incidents spanning 2–5 years. Average incident durations vary widely: GitHub averages 114 minutes, OpenAI 379 minutes, and Atlassian 1237 minutes. Each record has 15+ features including categorical (service name, impact level, weekday), numerical (duration in minutes, number of updates, component count), and temporal (hour, month, year) variables. The target variable for prediction is incident duration, which can be modeled as a regression task or classified into bins (short: <30min, medium: 30–120min, long: >120min).
