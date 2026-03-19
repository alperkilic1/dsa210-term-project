import json
import time
from datetime import datetime
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

SERVICES = {
    "github": "https://www.githubstatus.com/api/v2/incidents.json",
    "openai": "https://status.openai.com/api/v2/incidents.json",
    "cloudflare": "https://www.cloudflarestatus.com/api/v2/incidents.json",
    "twilio": "https://status.twilio.com/api/v2/incidents.json",
    "datadog": "https://status.datadoghq.com/api/v2/incidents.json",
    "atlassian": "https://status.atlassian.com/api/v2/incidents.json",
    "reddit": "https://www.redditstatus.com/api/v2/incidents.json",
    "discord": "https://discordstatus.com/api/v2/incidents.json",
    "dropbox": "https://status.dropbox.com/api/v2/incidents.json",
    "vercel": "https://www.vercel-status.com/api/v2/incidents.json",
    "netlify": "https://www.netlifystatus.com/api/v2/incidents.json",
    "heroku": "https://status.heroku.com/api/v2/incidents.json",
    "digitalocean": "https://status.digitalocean.com/api/v2/incidents.json",
    "linear": "https://linearstatus.com/api/v2/incidents.json",
    "notion": "https://status.notion.so/api/v2/incidents.json",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def fetch_incidents(name, url):
    all_incidents = []
    page = 1

    while True:
        try:
            resp = requests.get(f"{url}?page={page}&per_page=100", headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break

            incidents = resp.json().get("incidents", [])
            if not incidents:
                break

            all_incidents.extend(incidents)
            print(f"  {name} page {page}: {len(incidents)} incidents")

            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  {name} error: {e}")
            break

    return all_incidents


def parse_incident(incident, service):
    updates = incident.get("incident_updates", [])
    components = incident.get("components", [])

    created = incident.get("created_at", "")
    resolved = incident.get("resolved_at")

    duration = None
    if created and resolved:
        try:
            t1 = datetime.fromisoformat(created.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(resolved.replace("Z", "+00:00"))
            duration = round((t2 - t1).total_seconds() / 60, 1)
        except:
            pass

    return {
        "service": service,
        "id": incident.get("id", ""),
        "name": incident.get("name", ""),
        "status": incident.get("status", ""),
        "impact": incident.get("impact", "none"),
        "created_at": created,
        "resolved_at": resolved,
        "duration_minutes": duration,
        "num_updates": len(updates),
        "num_components": len(components),
        "components": [c.get("name", "") for c in components],
        "shortlink": incident.get("shortlink", ""),
        "first_update": updates[-1].get("body", "")[:200] if updates else "",
    }


def add_time_features(record):
    created = record.get("created_at", "")
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            record["created_hour"] = dt.hour
            record["created_weekday"] = dt.strftime("%A")
            record["created_date"] = dt.strftime("%Y-%m-%d")
            record["created_month"] = dt.month
            record["created_year"] = dt.year
        except:
            pass
    return record


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_parsed = []
    stats = {}

    for name, url in sorted(SERVICES.items()):
        print(f"\nCollecting {name}...")
        raw = fetch_incidents(name, url)

        if raw:
            raw_file = RAW_DIR / f"{name}_raw.json"
            raw_file.write_text(json.dumps(raw, indent=2, ensure_ascii=False))

            parsed = [add_time_features(parse_incident(inc, name)) for inc in raw]
            all_parsed.extend(parsed)

            durations = [p["duration_minutes"] for p in parsed if p["duration_minutes"] is not None]
            avg = round(sum(durations) / len(durations), 1) if durations else None
            stats[name] = {"total": len(parsed), "resolved": sum(1 for p in parsed if p["status"] == "resolved"), "avg_duration_min": avg}

            print(f"  {name}: {len(parsed)} incidents, avg duration: {avg}min")
        else:
            stats[name] = {"total": 0, "resolved": 0, "avg_duration_min": None}

        time.sleep(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "incidents.json").write_text(json.dumps(all_parsed, indent=2, ensure_ascii=False))
    (DATA_DIR / "stats.json").write_text(json.dumps({"collected_at": datetime.now().isoformat(), "total": len(all_parsed), "services": stats}, indent=2))

    print(f"\nDone! {len(all_parsed)} incidents from {len(stats)} services")


if __name__ == "__main__":
    main()
