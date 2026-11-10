"""Genera data.json con pronósticos de los próximos partidos.
Datos gratuitos de football-data.co.uk (resultados, remates, tarjetas y cuotas de varias casas)."""
import io, json, math, re, unicodedata, urllib.request, urllib.error, datetime as dt
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo

BASE = "https://www.football-data.co.uk/"
LEAGUES = {"SP1": "LaLiga", "SP2": "LaLiga 2", "E0": "Premier League", "E1": "Championship", "D1": "Bundesliga",
           "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "P1": "Primeira Liga",
           "B1": "Liga belga", "SC0": "Premiership escocesa", "T1": "Süper Lig", "G1": "Super League griega"}
COMP = {"uefa.champions": "Champions League", "uefa.europa": "Europa League", "uefa.europa.conf": "Conference League"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard?dates={}"  # fuente no oficial de ESPN
NAT = {"uefa.nations": "Nations League", "concacaf.nations.league": "Concacaf Nations League",
       "caf.nations_qual": "Clasificación Copa África", "fifa.friendly": "Amistosos", "fifa.world": "Mundial",
       "fifa.worldq.uefa": "Clasificación Mundial (UEFA)", "fifa.worldq.conmebol": "Clasificación Mundial (CONMEBOL)"}
MAD = ZoneInfo("Europe/Madrid")
WINDOW = 7  # días hacia delante
BOOKS = {"B365": "Bet365", "BW": "Betway", "IW": "Interwetten", "PS": "Pinnacle", "WH": "William Hill",
         "VC": "VC Bet", "BF": "Betfair", "LB": "Ladbrokes", "1XB": "1xBet"}
HALFLIFE = 150  # días: los partidos recientes pesan más

def get(url):
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    try: text = raw.decode("utf-8-sig")          # quita el carácter invisible (BOM) del principio
    except UnicodeDecodeError: text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("\ufeff").replace("ï»¿", "") for c in df.columns]
    return df

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

def rows(a, b, lines, tl):  # probabilidad de superar cada línea: por equipo y total del partido
    return {"lines": lines, "h": [round(over(a, x), 3) for x in lines], "a": [round(over(b, x), 3) for x in lines],
            "tl": tl, "t": [round(over(a + b, x), 3) for x in tl]}

def ou_odds(row):  # cuotas Más/Menos 2,5 goles de cada casa (columnas "B365>2.5" y "B365<2.5")
    out = {}
    for c in row.index:
        if str(c).endswith(">2.5"):
            p = str(c)[:-4]
            try: t = [float(row[p + ">2.5"]), float(row[p + "<2.5"])]
            except Exception: continue
            if all(v > 1 for v in t): out[p] = t
    return out

def am(x):  # cuota americana -> decimal
    try: v = float(str(x).replace("+", ""))
    except Exception: return None
    return round(1 + v / 100 if v > 0 else 1 + 100 / abs(v), 2) if v else None

def espn_odds(c):  # cuotas que ESPN publica en algunos partidos (suele ser una sola casa)
    for o in c.get("odds", []) or []:
        try: h, a = am(o["homeTeamOdds"]["moneyLine"]), am(o["awayTeamOdds"]["moneyLine"])
        except Exception: continue
        d = am((o.get("drawOdds") or {}).get("moneyLine"))
        if h and a: return {(o.get("provider") or {}).get("name", "ESPN"): [h, d, a] if d else [h, a]}
    return None

def build_match(r, mg, ms, mc, league=None, when=None, ext=None):
    d0, t0 = when or fd_when(r)
    lh, la = mg(r.HomeTeam, r.AwayTeam)
    M = np.outer(pois(lh, 12), pois(la, 12)); M /= M.sum()
    p = [float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())]
    m = {"league": league or LEAGUES[r.Div], "date": d0, "time": t0,
         "home": r.HomeTeam, "away": r.AwayTeam, "p": [round(x, 3) for x in p], "pick": int(np.argmax(p)),
         "goals": {"exp": round(lh + la, 2), "o25": round(over(lh + la, 2.5), 3),
                   "btts": round((1 - math.exp(-lh)) * (1 - math.exp(-la)), 3)}}
    m["tbl"] = {"goles": rows(lh, la, [0.5, 1.5, 2.5], [1.5, 2.5, 3.5])}
    if ms:
        a, b = ms(r.HomeTeam, r.AwayTeam)
        m["tbl"]["remates"] = rows(a, b, [1.5, 2.5, 3.5, 4.5], [6.5, 8.5, 10.5])
        m["sot"] = {"home": round(a, 1), "away": round(b, 1), "o85": round(over(a + b, 8.5), 3)}
    if mc:
        a, b = mc(r.HomeTeam, r.AwayTeam)
        m["tbl"]["tarjetas"] = rows(a, b, [0.5, 1.5, 2.5], [2.5, 3.5, 4.5])
        m["cards"] = {"home": round(a, 1), "away": round(b, 1), "o35": round(over(a + b, 3.5), 3), "o45": round(over(a + b, 4.5), 3)}
    o = odds(r)
    if o:
        best = [max(v[i] for v in o.values()) for i in range(3)]
        m["odds"] = {"books": {BOOKS.get(k, k): v for k, v in o.items() if k not in ("Max", "Avg")},
                     "best": best, "ev": [round(p[i] * best[i] - 1, 3) for i in range(3)]}
        m["value"] = [i for i in range(3) if m["odds"]["ev"][i] > 0.05]
    else: m["value"] = []
    if not o and ext and len(next(iter(ext.values()))) == 3:
        best = [max(v[i] for v in ext.values()) for i in range(3)]
        m["odds"] = {"books": ext, "best": best, "ev": [round(p[i] * best[i] - 1, 3) for i in range(3)]}
        m["value"] = [i for i in range(3) if m["odds"]["ev"][i] > 0.05]
    ou = ou_odds(r)
    if ou:
        names = {**BOOKS, "P": "Pinnacle"}
        best = [max(v[i] for v in ou.values()) for i in range(2)]
        po = m["goals"]["o25"]
        m["ou"] = {"books": {names.get(k, k): v for k, v in ou.items() if k not in ("Max", "Avg")}, "best": best,
                   "ev": [round(po * best[0] - 1, 3), round((1 - po) * best[1] - 1, 3)]}
    return m

def espn_events(path, d1, d2, step=10):  # path: "soccer/uefa.champions", "basketball/nba", "tennis/atp"
    evs, seen, errs, d = [], set(), [], d1
    while d <= d2:
        e = min(d + dt.timedelta(days=step - 1), d2)
        url = ESPN.format(path, f"{d:%Y%m%d}-{e:%Y%m%d}")
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
            got = json.loads(raw).get("events", [])
        except urllib.error.HTTPError as ex:
            got = []
            if step > 10:  # si el tramo largo falla, se reintenta en tramos de 10 días
                try: got = espn_events(path, d, e, 10)
                except Exception as ex2: errs.append(str(ex2))
            else: errs.append(f"HTTP {ex.code} {ex.read()[:120]!r} url={url}")
        for ev in got:
            if ev.get("id") not in seen: seen.add(ev.get("id")); evs.append(ev)
        d = e + dt.timedelta(days=1)
    if errs and not evs: raise RuntimeError(errs[0])
    return evs

def parse(ev):
    c = ev["competitions"][0]
    h = next(t for t in c["competitors"] if t["homeAway"] == "home")
    a = next(t for t in c["competitors"] if t["homeAway"] == "away")
    st = ev.get("status", {}).get("type", {})
    return dict(when=pd.Timestamp(ev["date"]), HomeTeam=h["team"]["displayName"], AwayTeam=a["team"]["displayName"],
                hs=h.get("score"), as_=a.get("score"), done=bool(st.get("completed")) or st.get("state") == "post", odds=espn_odds(c))

# --- UEFA Nations League (masculina), jornadas 5 y 6 (12-17 nov 2026). Horas de España (la fuente usa hora de México: +7 h).
# Fuente: calendario de Infobae con los grupos oficiales de la UEFA. Si ESPN funciona, sus datos prevalecen sobre esta lista.
_NL = {
 "2026-11-12": [("18:00", "Armenia", "Cyprus"), ("18:00", "Turkey", "Belgium"), ("20:45", "Italy", "France"), ("20:45", "Czechia", "Spain"),
                ("20:45", "Albania", "Finland"), ("20:45", "San Marino", "Belarus"), ("20:45", "England", "Croatia"), ("20:45", "Montenegro", "Latvia")],
 "2026-11-13": [("18:00", "Moldova", "Kazakhstan"), ("20:45", "Slovenia", "Switzerland"), ("20:45", "Netherlands", "Greece"),
                ("20:45", "Andorra", "Gibraltar"), ("20:45", "Luxembourg", "Estonia"), ("20:45", "Serbia", "Germany"),
                ("20:45", "Scotland", "North Macedonia"), ("20:45", "Slovakia", "Faroe Islands"), ("20:45", "Liechtenstein", "Azerbaijan"),
                ("20:45", "Bulgaria", "Iceland")],
 "2026-11-14": [("15:00", "Kosovo", "Israel"), ("18:00", "Georgia", "Hungary"), ("18:00", "Norway", "Wales"),
                ("20:45", "Austria", "Republic of Ireland"), ("20:45", "Portugal", "Denmark"), ("20:45", "Northern Ireland", "Ukraine"),
                ("20:45", "Sweden", "Bosnia and Herzegovina"), ("20:45", "Romania", "Poland")],
 "2026-11-15": [("15:00", "Latvia", "Armenia"), ("15:00", "Cyprus", "Montenegro"), ("18:00", "Finland", "San Marino"), ("18:00", "Belarus", "Albania"),
                ("20:45", "France", "Turkey"), ("20:45", "Belgium", "Italy"), ("20:45", "Croatia", "Czechia"), ("20:45", "Spain", "England")],
 "2026-11-16": [("16:00", "Kazakhstan", "Slovakia"), ("16:00", "Faroe Islands", "Moldova"), ("18:00", "Estonia", "Bulgaria"),
                ("18:00", "Lithuania", "Liechtenstein"), ("18:00", "Iceland", "Luxembourg"), ("20:45", "Germany", "Netherlands"),
                ("20:45", "Gibraltar", "Malta"), ("20:45", "Switzerland", "Scotland"), ("20:45", "North Macedonia", "Slovenia"), ("20:45", "Greece", "Serbia")],
 "2026-11-17": [("20:45", "Denmark", "Norway"), ("20:45", "Bosnia and Herzegovina", "Romania"), ("20:45", "Wales", "Portugal"),
                ("20:45", "Republic of Ireland", "Kosovo"), ("20:45", "Ukraine", "Georgia"), ("20:45", "Hungary", "Northern Ireland"),
                ("20:45", "Poland", "Sweden"), ("20:45", "Israel", "Austria")],
}
NL_MANUAL = [(d, t, h, a) for d, ms in _NL.items() for t, h, a in ms]
# Nivel aproximado de cada selección (estimación propia, NO es un dato oficial). Solo se usa si no hay resultados reales.
NL_ELO = {"Spain": 2150, "France": 2080, "England": 2030, "Portugal": 2000, "Netherlands": 1980, "Germany": 1960, "Italy": 1930, "Croatia": 1900,
          "Belgium": 1890, "Denmark": 1850, "Switzerland": 1840, "Austria": 1830, "Norway": 1830, "Turkey": 1800, "Serbia": 1760, "Scotland": 1760,
          "Ukraine": 1760, "Poland": 1750, "Czechia": 1750, "Hungary": 1730, "Sweden": 1730, "Greece": 1720, "Romania": 1700, "Wales": 1690,
          "Slovakia": 1650, "Republic of Ireland": 1650, "Bosnia and Herzegovina": 1650, "Slovenia": 1650, "Georgia": 1640, "Israel": 1620,
          "Albania": 1620, "Iceland": 1620, "North Macedonia": 1600, "Kosovo": 1600, "Northern Ireland": 1580, "Finland": 1570, "Montenegro": 1550,
          "Bulgaria": 1530, "Belarus": 1500, "Kazakhstan": 1500, "Armenia": 1480, "Cyprus": 1450, "Luxembourg": 1450, "Latvia": 1400,
          "Estonia": 1400, "Faroe Islands": 1400, "Lithuania": 1380, "Azerbaijan": 1380, "Moldova": 1350, "Malta": 1250, "San Marino": 1100,
          "Gibraltar": 1100, "Andorra": 1100, "Liechtenstein": 1100}
ALIAS = {"turkiye": "turkey", "czech republic": "czechia", "ireland": "republic of ireland", "bosnia-herzegovina": "bosnia and herzegovina"}

def norm(x):
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[^a-z]", "", ALIAS.get(x, x))

def elo_goals(h, a):
    x = (NL_ELO.get(h, 1700) + 70 - NL_ELO.get(a, 1700)) / 400  # +70 por jugar en casa
    return 1.35 * 2 ** x, 1.35 * 2 ** -x

def nl_manual(today, have):
    out = []
    for d, t, h, a in NL_MANUAL:
        if not (str(today) <= d <= str(today + dt.timedelta(days=WINDOW))) or (d, norm(h)[:6], norm(a)[:6]) in have: continue
        m = build_match(pd.Series({"HomeTeam": h, "AwayTeam": a}), elo_goals, None, None, league="Nations League", when=(d, t))
        m["cat"] = "selecciones"
        m["nota"] = "Pronóstico orientativo: el nivel de los equipos es una estimación a mano, sin resultados recientes."
        out.append(m)
    return out

def phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def madrid(ts):
    ts = ts if ts.tzinfo else ts.tz_localize("UTC")
    return ts.tz_convert(MAD)

def in_window(w, today): return today <= w.date() <= today + dt.timedelta(days=WINDOW)

def soccer_group(slugs, today, info, start, cat, label):  # solo modelo de goles (sin remates, tarjetas ni cuotas)
    evs = []
    for slug, name in slugs.items():
        try: espn_events("soccer/" + slug, today, today)  # comprobación rápida: ¿existe esta competición en ESPN?
        except Exception as e:
            info.append(f"{name}: no disponible ({str(e)[:90]})"); continue
        try: evs += [dict(parse(e), comp=name) for e in espn_events("soccer/" + slug, start, today + dt.timedelta(days=WINDOW), 30 if cat == "selecciones" else 10)]
        except Exception as e: info.append(f"{name}: error {e}")
    done = [e for e in evs if e["done"] and e["hs"] not in (None, "") and e["as_"] not in (None, "")]
    mg = None
    if done:
        df = pd.DataFrame(done)
        df["FTHG"] = pd.to_numeric(df.hs); df["FTAG"] = pd.to_numeric(df.as_)
        df["Date"] = df.when.dt.tz_convert(None)
        df["w"] = 0.5 ** ((pd.Timestamp(today) - df.Date).dt.days.clip(lower=0) / HALFLIFE)
        mg = model(df, "FTHG", "FTAG", 15)
    mg = mg or (lambda h, a: (1.5, 1.2))
    out = []
    for e in evs:
        w = madrid(e["when"])
        if e["done"] or not in_window(w, today): continue
        try:
            m = build_match(pd.Series({"HomeTeam": e["HomeTeam"], "AwayTeam": e["AwayTeam"]}), mg, None, None,
                            league=e["comp"], when=(w.strftime("%Y-%m-%d"), w.strftime("%H:%M")), ext=e.get("odds"))
            m["cat"] = cat; out.append(m)
        except Exception as ex: print("error", e["HomeTeam"], ex)
    info.append(f"{label}: {len(out)} partidos ({len(done)} resultados usados)")
    return out

def nba(today, info):
    evs = [parse(e) for e in espn_events("basketball/nba", today - dt.timedelta(days=300), today + dt.timedelta(days=WINDOW))]
    done = [e for e in evs if e["done"] and e["hs"] not in (None, "") and e["as_"] not in (None, "")]
    if len(done) < 40:
        info.append(f"NBA: pocos resultados ({len(done)})"); return []
    df = pd.DataFrame(done)
    df["FTHG"] = pd.to_numeric(df.hs); df["FTAG"] = pd.to_numeric(df.as_); df["Date"] = df.when.dt.tz_convert(None)
    df["w"] = 0.5 ** ((pd.Timestamp(today) - df.Date).dt.days.clip(lower=0) / HALFLIFE)
    mg, out = model(df, "FTHG", "FTAG", 40), []
    for e in evs:
        w = madrid(e["when"])
        if e["done"] or not in_window(w, today): continue
        eh, ea = mg(e["HomeTeam"], e["AwayTeam"]); tot = eh + ea; line = math.floor(tot) + 0.5
        ph = phi((eh - ea) / 12.5)  # el margen típico en la NBA varía unos 12,5 puntos
        mm = ({"cat": "basket", "league": "NBA", "date": w.strftime("%Y-%m-%d"), "time": w.strftime("%H:%M"),
                    "home": e["HomeTeam"], "away": e["AwayTeam"], "p": [round(ph, 3), round(1 - ph, 3)], "pick": 0 if ph >= .5 else 1,
                    "pts": {"home": round(eh, 1), "away": round(ea, 1), "total": round(tot, 1), "line": line,
                            "over": round(1 - phi((line - tot) / 19), 3)}, "value": []})
        ext = e.get("odds")
        if ext and len(next(iter(ext.values()))) == 2:
            best = [max(v[i] for v in ext.values()) for i in range(2)]
            mm["odds"] = {"books": ext, "best": best, "ev": [round(mm["p"][i] * best[i] - 1, 3) for i in range(2)]}
        out.append(mm)
    info.append(f"NBA: {len(out)} partidos ({len(done)} resultados usados)")
    return out

def tennis(today, info):  # Elo sencillo con los resultados de los últimos 150 días
    out = []
    for tour, name in (("atp", "ATP"), ("wta", "WTA")):
        try: evs = espn_events(f"tennis/{tour}", today - dt.timedelta(days=150), today + dt.timedelta(days=WINDOW))
        except Exception as e:
            info.append(f"{name}: error {e}"); continue
        ms, seen = [], set()
        for ev in evs:
            for g in ev.get("groupings", []):
                if "singles" not in str((g.get("grouping") or {}).get("slug", "")): continue
                for c in g.get("competitions", []):
                    cp = c.get("competitors", [])
                    if c.get("id") in seen or len(cp) != 2: continue
                    seen.add(c.get("id"))
                    try: names = [x["athlete"]["displayName"] for x in cp]
                    except Exception: continue
                    st = (c.get("status") or {}).get("type", {})
                    ms.append(dict(when=pd.Timestamp(c.get("date") or ev["date"]), names=names, win=[bool(x.get("winner")) for x in cp],
                                   done=bool(st.get("completed")) or st.get("state") == "post"))
        ms.sort(key=lambda x: x["when"])
        R, N = {}, {}
        for m in ms:
            if m["done"] and sum(m["win"]) == 1:
                a, b = m["names"]; ra, rb = R.get(a, 1500.0), R.get(b, 1500.0)
                pa = 1 / (1 + 10 ** ((rb - ra) / 400)); d = 32 * ((1.0 if m["win"][0] else 0.0) - pa)
                R[a], R[b] = ra + d, rb - d; N[a] = N.get(a, 0) + 1; N[b] = N.get(b, 0) + 1
        n0 = len(out)
        for m in ms:
            w = madrid(m["when"])
            if m["done"] or not in_window(w, today): continue
            a, b = m["names"]
            if "TBD" in (a, b): continue
            ra, rb = R.get(a, 1500.0), R.get(b, 1500.0); pa = 1 / (1 + 10 ** ((rb - ra) / 400))
            out.append({"cat": "tenis", "league": name, "date": w.strftime("%Y-%m-%d"), "time": w.strftime("%H:%M"), "home": a, "away": b,
                        "p": [round(pa, 3), round(1 - pa, 3)], "pick": 0 if pa >= .5 else 1,
                        "elo": [round(ra), round(rb)], "n": [N.get(a, 0), N.get(b, 0)], "value": []})
        info.append(f"{name}: {len(out) - n0} partidos ({sum(1 for m in ms if m['done'])} resultados usados)")
    return out

def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o

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
    start = dt.date(today.year if today.month >= 7 else today.year - 1, 8, 15)
    for fn, args in ((soccer_group, (COMP, today, info, start, "futbol", "Clubes UEFA")),
                     (soccer_group, (NAT, today, info, today - dt.timedelta(days=365), "selecciones", "Selecciones")),
                     (nba, (today, info)), (tennis, (today, info))):
        try: out += fn(*args)
        except Exception as e: info.append(f"{fn.__name__}: error {e}")
    have = {(m["date"], norm(m["home"])[:6], norm(m["away"])[:6]) for m in out if m.get("cat") == "selecciones"}
    extra = nl_manual(today, have)
    out += extra
    info.append(f"Nations League (lista manual): {len(extra)} partidos añadidos")
    out = [m for m in out if str(today) <= m["date"] <= str(end)]
    out.sort(key=lambda m: (m["date"], m["time"]))
    if not out and any("error" in i for i in info):
        print(info); raise SystemExit("Sin datos: fallaron las fuentes")  # el workflow falla y se conserva el data.json anterior
    try: old = json.load(open("data.json", encoding="utf-8"))
    except Exception: old = {}
    same = json.dumps(old.get("matches"), sort_keys=True) == json.dumps(out, sort_keys=True)
    upd = old["updated"] if same and old.get("updated") else dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    json.dump(clean({"updated": upd, "info": info, "matches": out}), open("data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, allow_nan=False)
    print(len(out), "partidos", info)

if __name__ == "__main__":
    main()
