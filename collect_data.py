import json
import time
from datetime import datetime
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

SERVICES = {
    "github": "https://www.githubstatus.com",
    "openai": "https://status.openai.com",
    "cloudflare": "https://www.cloudflarestatus.com",
    "twilio": "https://status.twilio.com",
    "datadog": "https://status.datadoghq.com",
    "atlassian": "https://status.atlassian.com",
    "reddit": "https://www.redditstatus.com",
    "discord": "https://discordstatus.com",
    "dropbox": "https://status.dropbox.com",
    "vercel": "https://www.vercel-status.com",
    "netlify": "https://www.netlifystatus.com",
    "heroku": "https://status.heroku.com",
    "digitalocean": "https://status.digitalocean.com",
    "linear": "https://linearstatus.com",
    "notion": "https://status.notion.so",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

MAX_HISTORY_PAGES = 3
MAX_EXTRA_PER_SERVICE = 60


def fetch_incidents(name, base_url):
    recent = _fetch_recent(name, base_url)
    seen_ids = {inc["id"] for inc in recent}

    history_items = _fetch_history_items(name, base_url)
    new_items = [(c, h) for c, h in history_items if c not in seen_ids]
    to_fetch = new_items[:MAX_EXTRA_PER_SERVICE]
    print(f"  {name}: {len(recent)} recent, {len(new_items)} in history, fetching {len(to_fetch)} extra")

    extra = _fetch_by_codes_with_fallback(name, base_url, to_fetch)
    all_incidents = recent + extra
    print(f"  {name}: {len(all_incidents)} total")
    return all_incidents


def _fetch_recent(name, base_url):
    try:
        url = f"{base_url}/api/v2/incidents.json"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("incidents", [])
    except Exception as e:
        print(f"  {name} recent fetch error: {e}")
    return []


def _fetch_history_items(name, base_url):
    items = []
    for page in range(1, MAX_HISTORY_PAGES + 1):
        try:
            url = f"{base_url}/history.json?page={page}"
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break
            months = resp.json().get("months", [])
            page_items = []
            for month in months:
                for inc in month.get("incidents", []):
                    code = inc.get("code", "")
                    if code:
                        page_items.append((code, inc))
            if not page_items:
                break
            items.extend(page_items)
            time.sleep(0.3)
        except Exception as e:
            print(f"  {name} history page {page} error: {e}")
            break
    return items


def _history_to_incident(code, hist):
    return {
        "id": code,
        "name": hist.get("name", ""),
        "status": "resolved",
        "impact": hist.get("impact", "none"),
        "created_at": "",
        "resolved_at": None,
        "incident_updates": [],
        "components": [],
        "shortlink": "",
        "_from_history": True,
    }


def _fetch_by_codes_with_fallback(name, base_url, items):
    results = []
    api_ok = 0
    api_fail = 0
    for i, (code, hist) in enumerate(items):
        try:
            url = f"{base_url}/api/v2/incidents/{code}.json"
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                inc = resp.json().get("incident")
                if inc:
                    results.append(inc)
                    api_ok += 1
                else:
                    results.append(_history_to_incident(code, hist))
                    api_fail += 1
            else:
                results.append(_history_to_incident(code, hist))
                api_fail += 1
            if (i + 1) % 20 == 0:
                print(f"    {name}: fetched {i + 1}/{len(items)} (api={api_ok}, fallback={api_fail})")
            time.sleep(0.15)
        except Exception as e:
            results.append(_history_to_incident(code, hist))
            api_fail += 1
            print(f"  {name} incident {code} error: {e}")
    if api_fail > 0:
        print(f"    {name}: {api_fail} used history fallback (no detailed timestamps)")
    return results


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
