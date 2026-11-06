"""Genera data.json con pronósticos de los próximos partidos.
Datos gratuitos de football-data.co.uk (resultados, remates, tarjetas y cuotas de varias casas)."""
import io, json, math, re, urllib.request, datetime as dt
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo

BASE = "https://www.football-data.co.uk/"
LEAGUES = {"SP1": "LaLiga", "SP2": "LaLiga 2", "E0": "Premier League", "E1": "Championship", "D1": "Bundesliga",
           "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "P1": "Primeira Liga",
           "B1": "Liga belga", "SC0": "Premiership escocesa", "T1": "Süper Lig", "G1": "Super League griega"}
COMP = {"uefa.champions": "Champions League", "uefa.europa": "Europa League", "uefa.europa.conf": "Conference League"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/scoreboard?limit=1000&dates={}"  # fuente no oficial de ESPN
MAD = ZoneInfo("Europe/Madrid")
WINDOW = 7  # días hacia delante
BOOKS = {"B365": "Bet365", "BW": "Betway", "IW": "Interwetten", "PS": "Pinnacle", "WH": "William Hill",
         "VC": "VC Bet", "BF": "Betfair", "LB": "Ladbrokes", "1XB": "1xBet"}
HALFLIFE = 150  # días: los partidos recientes pesan más

def get(url):
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")

def history(div, today):
    y = today.year if today.month >= 7 else today.year - 1
    frames = []
    for s in (f"{(y-1)%100:02d}{y%100:02d}", f"{y%100:02d}{(y+1)%100:02d}"):
        try: frames.append(get(f"{BASE}mmz4281/{s}/{div}.csv"))
        except Exception as e: print("sin datos", div, s, e)
    if not frames: return None
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ["FTHG", "FTAG", "HST", "AST", "HY", "AY", "HR", "AR"]:
        df[c] = pd.to_numeric(df[c] if c in df else pd.Series(np.nan, index=df.index), errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()
    df["HC"] = df.HY + df.HR.fillna(0); df["AC"] = df.AY + df.AR.fillna(0)
    df["w"] = 0.5 ** ((pd.Timestamp(today) - df.Date).dt.days.clip(lower=0) / HALFLIFE)
    return df

def wm(x, w, prior, k=3.0):  # media ponderada encogida hacia la media de la liga
    return (float((x * w).sum()) + k * prior) / (float(w.sum()) + k)

def model(df, hc, ac, minn=30):
    d = df.dropna(subset=[hc, ac])
    if len(d) < minn: return None
    ah = float((d[hc] * d.w).sum() / d.w.sum()); aa = float((d[ac] * d.w).sum() / d.w.sum())
    def exp(home, away):
        H, A = d[d.HomeTeam == home], d[d.AwayTeam == away]
        fh, ch = wm(H[hc], H.w, ah), wm(H[ac], H.w, aa)   # local: marca / concede en casa
        fa, ca = wm(A[ac], A.w, aa), wm(A[hc], A.w, ah)   # visitante: marca / concede fuera
        return fh * ca / ah, fa * ch / aa
    return exp

def pois(l, n=40): return np.array([math.exp(-l) * l ** k / math.factorial(k) for k in range(n)])
def over(l, line): return float(pois(l)[int(line) + 1:].sum())

def odds(row):
    out = {}
    for p in list(BOOKS) + ["Max", "Avg"]:
        try: t = [float(row[p + x]) for x in "HDA"]
        except Exception: continue
        if all(v > 1 for v in t): out[p] = t
    return out

def fd_when(r):  # football-data da la hora de Reino Unido: la pasamos a hora de España
    t = str(r.get("Time", "") or "")
    try:
        ts = pd.Timestamp(f"{r.Date:%Y-%m-%d} {t}").tz_localize("Europe/London").tz_convert(MAD)
        return ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")
    except Exception:
        return r.Date.strftime("%Y-%m-%d"), t

def build_match(r, mg, ms, mc, league=None, when=None):
    d0, t0 = when or fd_when(r)
    lh, la = mg(r.HomeTeam, r.AwayTeam)
    M = np.outer(pois(lh, 12), pois(la, 12)); M /= M.sum()
    p = [float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())]
    m = {"league": league or LEAGUES[r.Div], "date": d0, "time": t0,
         "home": r.HomeTeam, "away": r.AwayTeam, "p": [round(x, 3) for x in p], "pick": int(np.argmax(p)),
         "goals": {"exp": round(lh + la, 2), "o25": round(over(lh + la, 2.5), 3),
                   "btts": round((1 - math.exp(-lh)) * (1 - math.exp(-la)), 3)}}
    if ms:
        a, b = ms(r.HomeTeam, r.AwayTeam)
        m["sot"] = {"home": round(a, 1), "away": round(b, 1), "o85": round(over(a + b, 8.5), 3)}
    if mc:
        a, b = mc(r.HomeTeam, r.AwayTeam)
        m["cards"] = {"home": round(a, 1), "away": round(b, 1), "o35": round(over(a + b, 3.5), 3), "o45": round(over(a + b, 4.5), 3)}
    o = odds(r)
    if o:
        best = [max(v[i] for v in o.values()) for i in range(3)]
        m["odds"] = {"books": {BOOKS.get(k, k): v for k, v in o.items() if k not in ("Max", "Avg")},
                     "best": best, "ev": [round(p[i] * best[i] - 1, 3) for i in range(3)]}
        m["value"] = [i for i in range(3) if m["odds"]["ev"][i] > 0.05]
    else: m["value"] = []
    return m

def espn_events(slug, d1, d2):
    url = ESPN.format(slug, f"{d1:%Y%m%d}-{d2:%Y%m%d}")
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    return json.loads(raw).get("events", [])

def parse(ev):
    c = ev["competitions"][0]
    h = next(t for t in c["competitors"] if t["homeAway"] == "home")
    a = next(t for t in c["competitors"] if t["homeAway"] == "away")
    st = ev.get("status", {}).get("type", {})
    return dict(when=pd.Timestamp(ev["date"]), HomeTeam=h["team"]["displayName"], AwayTeam=a["team"]["displayName"],
                hs=h.get("score"), as_=a.get("score"), done=bool(st.get("completed")) or st.get("state") == "post")

def uefa(today, info):  # Champions / Europa / Conference: solo modelo de goles (sin remates, tarjetas ni cuotas)
    out = []
    start = dt.date(today.year if today.month >= 7 else today.year - 1, 7, 1)
    for slug, name in COMP.items():
        try: evs = [parse(e) for e in espn_events(slug, start, today + dt.timedelta(days=WINDOW))]
        except Exception as e:
            info.append(f"{name}: error {e}"); continue
        done = [e for e in evs if e["done"] and e["hs"] not in (None, "") and e["as_"] not in (None, "")]
        mg = None
        if done:
            df = pd.DataFrame(done)
            df["FTHG"] = pd.to_numeric(df.hs); df["FTAG"] = pd.to_numeric(df.as_)
            df["Date"] = df.when.dt.tz_convert(None)
            df["w"] = 0.5 ** ((pd.Timestamp(today) - df.Date).dt.days.clip(lower=0) / HALFLIFE)
            mg = model(df, "FTHG", "FTAG", 15)
        mg = mg or (lambda h, a: (1.5, 1.2))
        n = 0
        for e in evs:
            w = e["when"].tz_convert(MAD)
            if e["done"] or not (today <= w.date() <= today + dt.timedelta(days=WINDOW)): continue
            try:
                out.append(build_match(pd.Series({"HomeTeam": e["HomeTeam"], "AwayTeam": e["AwayTeam"]}), mg, None, None,
                                       league=name, when=(w.strftime("%Y-%m-%d"), w.strftime("%H:%M")))); n += 1
            except Exception as ex: print("error", e["HomeTeam"], ex)
        info.append(f"{name}: {n} partidos ({len(done)} resultados usados)")
    return out

def main():
    today = dt.date.today(); end = today + dt.timedelta(days=WINDOW); info, out = [], []
    try:
        fx = get(BASE + "fixtures.csv")
        fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
        fx = fx[fx.Div.isin(LEAGUES) & (fx.Date >= pd.Timestamp(today)) & (fx.Date <= pd.Timestamp(end))]
        info.append(f"Ligas nacionales (football-data.co.uk): {len(fx)} partidos en el rango")
    except Exception as e:
        fx = pd.DataFrame(columns=["Div"]); info.append(f"Ligas nacionales: error {e}")
    for div, g in fx.groupby("Div"):
        df = history(div, today)
        if df is None: continue
        mg, ms, mc = model(df, "FTHG", "FTAG"), model(df, "HST", "AST"), model(df, "HC", "AC")
        if not mg: continue
        for _, r in g.iterrows():
            try: out.append(build_match(r, mg, ms, mc))
            except Exception as e: print("error", r.HomeTeam, r.AwayTeam, e)
    out += uefa(today, info)
    out = [m for m in out if str(today) <= m["date"] <= str(end)]
    out.sort(key=lambda m: (m["date"], m["time"]))
    if not out and any("error" in i for i in info):
        print(info); raise SystemExit("Sin datos: fallaron las fuentes")  # el workflow falla y se conserva el data.json anterior
    try: old = json.load(open("data.json", encoding="utf-8"))
    except Exception: old = {}
    same = json.dumps(old.get("matches"), sort_keys=True) == json.dumps(out, sort_keys=True)
    upd = old["updated"] if same and old.get("updated") else dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    json.dump({"updated": upd, "info": info, "matches": out}, open("data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(out), "partidos", info)

if __name__ == "__main__":
    main()
