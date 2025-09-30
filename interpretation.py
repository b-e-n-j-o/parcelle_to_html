# -*- coding: utf-8 -*-
"""
Interpretation du rapport d'intersections (JSON) -> HTML lisible.

- Charge le JSON de sortie d'intersections (contrat immuable).
- Charge le catalogue pour récupérer: 'name' (nom humain) et 'keep' (colonnes utiles).
- Produit un HTML joliment formaté (Bootstrap).
- Déduplication: fusionne les lignes d'une couche si mêmes valeurs keep ET même surface (arrondie 0.1 m²).
- Masque les colonnes meta/id dans le tableau, mais liste les IDs associés sous forme repliable.
- Affiche en haut les bâtiments détectés via WFS IGN.
"""

import json
import os
from collections import defaultdict

# ================== CHEMINS EN DUR ==================
JSON_IN  = "/CONFIG/rapport_parcelle.json"
MAP_PATH = "/CONFIG/nouveau_catalogue_29_09.json"
HTML_OUT = "/rapport_parcelle.html"

# Colonnes à masquer dans les tableaux mais à remonter en "IDs associés"
ID_LIKE = {"id", "gid", "gml_id", "uid"}
# Champs meta à ne pas afficher en colonnes
META_SKIP = {"update", "source"}

def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_layer_meta(catalog, schema, table):
    fq = f"{schema}.{table}"
    meta = catalog.get(fq, {}) if isinstance(catalog, dict) else {}
    name = meta.get("name") or fq
    keep = list(meta.get("keep") or [])
    # Nettoyage keep: on retire IDs et META
    keep_clean = [c for c in keep if c not in ID_LIKE and c not in META_SKIP]
    return name, keep_clean, keep  # keep original aussi (pour lister IDs si présents)

def fmt_area(v):
    if v is None:
        return "—"
    return f"{round(float(v), 1):,.1f}".replace(",", " ")

def fmt_pct(v):
    if v is None:
        return "—"
    return f"{round(float(v), 2):.2f}%"

def dedup_rows(rows, keep_cols):
    """
    rows: liste de dicts {"values": {...}, "inter_area_m2": float, "pct": float, "ids": [..]}
    Fusionne si mêmes valeurs keep + même surface (arrondie 0.1 m²).
    """
    bucket = {}
    for r in rows:
        key_vals = tuple(
            (col, tuple(sorted(r["values"].get(col, []))) if isinstance(r["values"].get(col), list) else r["values"].get(col))
            for col in keep_cols
        )
        area_key = round(r.get("inter_area_m2") or 0.0, 1)
        key = (key_vals, area_key)
        if key not in bucket:
            bucket[key] = {
                "values": r["values"],
                "inter_area_m2": r.get("inter_area_m2"),
                "pct": r.get("pct"),
                "ids": set(r.get("ids") or []),
                "id_map": r.get("id_map") or defaultdict(list),
            }
        else:
            bucket[key]["ids"] |= set(r.get("ids") or [])
            for k, lst in (r.get("id_map") or {}).items():
                bucket[key]["id_map"][k].extend(lst)
    out = []
    for v in bucket.values():
        v["id_map"] = {k: sorted(set(lst)) for k, lst in v["id_map"].items()}
        v["ids"] = sorted(map(str, v["ids"]))
        out.append(v)
    return out

def build_layer_rows(layer_entry, keep_cols, keep_all_cols):
    n = int(layer_entry.get("count") or 0)
    if n <= 0 or not layer_entry.get("rows"):
        return []

    rows = []
    for r in layer_entry.get("rows"):
        row_vals = {c: r.get(c) for c in keep_cols}
        inter_area = r.get("inter_area_m2")
        pct = r.get("pct_of_parcel")
        one = {
            "values": row_vals,
            "inter_area_m2": inter_area,
            "pct": pct,
            "ids": [],
            "id_map": {},
        }
        # Collecte des ID-like
        flat_ids = []
        id_map = {}
        for k in keep_all_cols:
            if k in ID_LIKE and k in r:
                val = r[k]
                if val:
                    flat_ids.append(val)
                    id_map.setdefault(k, []).append(val)
        one["ids"] = flat_ids
        one["id_map"] = id_map
        rows.append(one)

    return dedup_rows(rows, keep_cols)

def html_escape(s):
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            )

def render_html(report, catalog):
    rep = report.get("reports", [{}])[0]
    parcel = rep.get("parcel", {})
    parcel_label = parcel.get("label", "—")
    layers = rep.get("results", [])
    layers_with_hits = int(rep.get("layers_with_hits") or 0)

    total_layers = len(layers)

    out = []
    out.append("""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport parcellaire</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { padding: 24px; }
  h1 { font-size: 1.6rem; margin-bottom: .25rem; }
  .subtitle { color:#666; margin-bottom: 1rem; }
  .layer-title { margin-top: 1.5rem; }
  table { font-size: 0.94rem; }
  th, td { vertical-align: middle; }
  .muted { color:#6c757d; }
  .small { font-size: .9rem; }
  .id-block { background:#f8f9fa; border:1px solid #e9ecef; border-radius:.5rem; padding:.5rem .75rem; }
</style>
</head>
<body>
""")

    out.append(f"<h1>Rapport d’intersections — parcelle <strong>{html_escape(parcel_label)}</strong></h1>")
    out.append(f"""<div class="subtitle">
Commune: <strong>{html_escape(report.get('commune', ''))}</strong> (INSEE {html_escape(report.get('insee',''))}) — Département: <strong>{html_escape(report.get('departement',''))}</strong><br>
Couches intersectées: <strong>{layers_with_hits}</strong> / {total_layers}
</div>""")

    # Bloc bâtiments
    buildings = rep.get("buildings") or []
    if buildings:
        out.append('<div class="mt-3 mb-3 p-3 border rounded bg-light">')
        out.append("<h2>Bâtiments présents sur la parcelle</h2>")

        # Résumé rapide
        types = [b.get("type") or "—" for b in buildings]
        type_counts = {t: types.count(t) for t in set(types)}
        out.append(f"<p><strong>{len(buildings)}</strong> bâtiment(s) détecté(s). "
                   + ", ".join(f"{n}× {t}" for t, n in type_counts.items()) + "</p>")

        # Détail liste
        out.append("<ul>")
        for b in buildings:
            type_b = b.get("type") or "—"
            out.append(f"<li><strong>{html_escape(type_b)}</strong></li>")
        out.append("</ul></div>")

    # Boucle couches
    for layer in layers:
        schema = layer.get("schema", "")
        table = layer.get("table", "")
        name, keep_cols, keep_all_cols = get_layer_meta(catalog, schema, table)

        out.append(f'<h2 class="layer-title">{html_escape(name)} <span class="muted">({html_escape(schema)}.{html_escape(table)})</span></h2>')

        rows = build_layer_rows(layer, keep_cols, keep_all_cols)

        if not rows:
            out.append('<div class="text-muted">Aucune intersection.</div>')
            continue

        out.append('<div class="table-responsive"><table class="table table-striped align-middle">')
        out.append("<thead><tr>")
        for c in keep_cols:
            out.append(f"<th>{html_escape(c)}</th>")
        out.append("<th>Surface (m²)</th><th>% parcelle</th>")
        out.append("</tr></thead><tbody>")

        for r in rows:
            out.append("<tr>")
            for c in keep_cols:
                out.append(f"<td>{html_escape(r['values'].get(c, '—') or '—')}</td>")
            out.append(f"<td>{fmt_area(r.get('inter_area_m2'))}</td>")
            out.append(f"<td>{fmt_pct(r.get('pct'))}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")

        # IDs associés
        any_ids = any(r.get("ids") for r in rows) or any(r.get("id_map") for r in rows)
        if any_ids:
            fused_map = defaultdict(set)
            for r in rows:
                for k, lst in (r.get("id_map") or {}).items():
                    fused_map[k].update(lst)
            fused_map = {k: sorted(v) for k, v in fused_map.items() if k in ID_LIKE and v}
            if not fused_map:
                all_ids = set()
                for r in rows:
                    all_ids.update(r.get("ids") or [])
                if all_ids:
                    fused_map = {"id": sorted(all_ids)}
            if fused_map:
                out.append("""
<details class="mt-2">
  <summary class="small">IDs associés</summary>
  <div class="id-block mt-2">""")
                for k in sorted(fused_map.keys()):
                    ids_str = ", ".join(html_escape(x) for x in fused_map[k][:200])
                    out.append(f"<div><strong>{html_escape(k)}</strong>: {ids_str}</div>")
                out.append("</div></details>")

    out.append("""
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body></html>""")
    return "\n".join(out)

def main():
    if not os.path.exists(JSON_IN):
        raise SystemExit(f"JSON introuvable: {JSON_IN}")
    if not os.path.exists(MAP_PATH):
        raise SystemExit(f"Catalogue introuvable: {MAP_PATH}")

    catalog = load_catalog(MAP_PATH)
    report = load_report(JSON_IN)

    html = render_html(report, catalog)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Rapport HTML écrit: {HTML_OUT}")

if __name__ == "__main__":
    main()
