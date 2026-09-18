"""
Generates one JSON file per day with a Vedic (Rashi-based) daily horoscope
for all 12 signs.

Pipeline:
  1. Compute today's real sidereal planetary positions (Lahiri ayanamsha)
     using Swiss Ephemeris (Moshier mode -- no ephemeris data files needed).
  2. Apply classical Gochara (transit) rules to work out, for each of the
     12 Rashis, which planets are favorably/unfavorably placed today.
  3. Send that structured, real data to Gemini and ask it to turn it into
     short, readable prose per sign -- Gemini writes the sentences, it
     does NOT invent the astrology.
  4. Write everything to docs/horoscope/latest.json (served by GitHub
     Pages) and archive a dated copy under docs/horoscope/archive/.

Run with:  python scripts/generate_daily_horoscope.py
Requires:  GEMINI_API_KEY environment variable.
"""

import os
import sys
import json
import datetime
import re

import swisseph as swe
import requests

# ---------------------------------------------------------------------------
# 1. Rashi + planet setup
# ---------------------------------------------------------------------------

RASHIS = [
    "Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
    "Tula", "Vrishchika", "Dhanu", "Makara", "Kumbha", "Meena",
]
RASHI_ENGLISH = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,  # Ketu is always 180 degrees opposite Rahu
}

# Classical Gochara (transit) house-distances traditionally considered
# favorable, counted inclusively from the reference Rashi (1 = same sign).
# These follow the commonly-cited rules from traditional Vedic transit
# astrology; some schools vary slightly -- this is a reasonable, widely
# used baseline, not a single universally-agreed standard.
GOCHARA_FAVORABLE = {
    "Sun": {3, 6, 10, 11},
    "Moon": {1, 3, 6, 7, 10, 11},
    "Mars": {3, 6, 11},
    "Mercury": {2, 4, 6, 8, 10, 11},
    "Jupiter": {2, 5, 7, 9, 11},
    "Venus": {1, 2, 3, 4, 5, 8, 9, 11, 12},
    "Saturn": {3, 6, 11},
    "Rahu": {3, 6, 10, 11},
    "Ketu": {3, 6, 10, 11},
}

SADE_SATI_HOUSES = {12, 1, 2}  # Saturn in these houses from Moon sign


def get_sidereal_longitude(jd_ut: float, planet_id: int) -> float:
    flags = swe.FLG_SIDEREAL | swe.FLG_MOSEPH
    result, _ = swe.calc_ut(jd_ut, planet_id, flags)
    return result[0]  # ecliptic longitude in degrees


def rashi_index_from_longitude(longitude: float) -> int:
    return int(longitude % 360 // 30)  # 0 = Mesha ... 11 = Meena


def house_distance(from_rashi_idx: int, to_rashi_idx: int) -> int:
    """1-12 inclusive count from from_rashi to to_rashi."""
    return ((to_rashi_idx - from_rashi_idx) % 12) + 1


def compute_today_transits(today: datetime.date) -> dict:
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    jd_ut = swe.julday(today.year, today.month, today.day, 0.0)  # 00:00 UT reference

    planet_rashi = {}
    for name, pid in PLANETS.items():
        lon = get_sidereal_longitude(jd_ut, pid)
        planet_rashi[name] = rashi_index_from_longitude(lon)
    # Ketu = Rahu + 180 degrees -> opposite Rashi
    planet_rashi["Ketu"] = (planet_rashi["Rahu"] + 6) % 12

    return planet_rashi


def build_transit_summary(planet_rashi: dict) -> list:
    """For each Rashi, work out which planets are favorable/unfavorable
    today per classical Gochara rules, plus a simple net tenor and a
    Sade Sati flag for Saturn."""
    summary = []
    for idx, rashi in enumerate(RASHIS):
        favorable, unfavorable = [], []
        sade_sati = False
        for planet, favorable_houses in GOCHARA_FAVORABLE.items():
            planet_idx = planet_rashi[planet]
            distance = house_distance(idx, planet_idx)
            if planet == "Saturn" and distance in SADE_SATI_HOUSES:
                sade_sati = True
            if distance in favorable_houses:
                favorable.append(planet)
            else:
                unfavorable.append(planet)

        net = len(favorable) - len(unfavorable)
        if net >= 2:
            tenor = "very favorable"
        elif net == 1:
            tenor = "favorable"
        elif net == 0:
            tenor = "neutral / mixed"
        elif net == -1:
            tenor = "mildly challenging"
        else:
            tenor = "challenging"

        summary.append({
            "rashi": rashi,
            "rashi_english": RASHI_ENGLISH[idx],
            "favorable_planets": favorable,
            "unfavorable_planets": unfavorable,
            "net_tenor": tenor,
            "sade_sati": sade_sati,
        })
    return summary


# ---------------------------------------------------------------------------
# 2. Gemini: turn the real computed data into readable prose
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def build_prompt(today: datetime.date, transit_summary: list) -> str:
    data_block = json.dumps(transit_summary, indent=2)
    return f"""You are writing a short daily Vedic astrology horoscope for an Indian
Panchang app, for {today.isoformat()}.

Below is REAL computed astrological transit data for all 12 Rashis (Vedic
moon signs), based on classical Gochara (transit) rules. Do not invent or
contradict this data -- only turn it into natural, readable prose.

{data_block}

For EACH of the 12 signs, write 2-3 short sentences (about 40-60 words) in
a warm, grounded, practical tone. Reflect the given "net_tenor" honestly
(don't make a "challenging" day sound falsely rosy, and don't make a
"very favorable" day sound alarming). If sade_sati is true, mention it
briefly and neutrally as a longer Saturn transit phase, not as something
frightening. Avoid specific medical, legal, or financial predictions or
instructions -- keep it general (mood, relationships, focus, energy).

Respond with ONLY valid JSON, no markdown code fences, no extra text, in
exactly this shape:

{{
  "Mesha": "text...",
  "Vrishabha": "text...",
  "Mithuna": "text...",
  "Karka": "text...",
  "Simha": "text...",
  "Kanya": "text...",
  "Tula": "text...",
  "Vrishchika": "text...",
  "Dhanu": "text...",
  "Makara": "text...",
  "Kumbha": "text...",
  "Meena": "text..."
}}"""


def call_gemini(prompt: str, api_key: str) -> dict:
    resp = requests.post(
        f"{GEMINI_URL}?key={api_key}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    # Strip accidental code fences just in case.
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)

# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    today = datetime.date.today()
    planet_rashi = compute_today_transits(today)
    transit_summary = build_transit_summary(planet_rashi)
    prompt = build_prompt(today, transit_summary)

    try:
        horoscope_text = call_gemini(prompt, api_key)
    except Exception as e:
        print(f"ERROR: Gemini call/parse failed: {e}", file=sys.stderr)
        sys.exit(1)  # Fail loudly -- do NOT overwrite yesterday's good file with a bad one.

    output = {
        "date": today.isoformat(),
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "ayanamsha": "Lahiri",
        "source": "Swiss Ephemeris (Moshier) transits + classical Gochara rules; prose by Gemini",
        "signs": [
            {
                **transit_summary[i],
                "horoscope": horoscope_text.get(RASHIS[i], ""),
            }
            for i in range(12)
        ],
    }

    os.makedirs("docs/horoscope/archive", exist_ok=True)
    with open("docs/horoscope/latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(f"docs/horoscope/archive/{today.isoformat()}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote horoscope for {today.isoformat()}.")


if __name__ == "__main__":
    main()
