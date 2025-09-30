# -*- coding: utf-8 -*-
"""
enclaves.py — Détection et soustraction d'enclaves pour une parcelle cadastrale IGN.

- Prend une feature WFS (GeoJSON) de parcelle ("host_feature").
- Récupère les parcelles voisines via WFS IGN dans la BBOX d’un buffer métrique (L93).
- Détecte les "enclaves" (parcelles entièrement à l'intérieur de l'hôte, sans toucher son contour).
- Soustrait ces enclaves de l'hôte et renvoie la géométrie corrigée (WKT/GeoJSON) + métriques.

Dépendances: shapely>=2, pyproj, requests
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import requests

from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection, mapping
from shapely.ops import unary_union, transform, snap
from shapely.validation import make_valid

from pyproj import Transformer

IGN_WFS = "https://data.geopf.fr/wfs/ows"
LAYER_PARCELLE = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"

# ---------- Utils géométrie ----------

def _to_polygonal(geom):
    """
    Force une géométrie Shapely en (Multi)Polygon en conservant les anneaux intérieurs.
    - Polygon  : conservé tel quel (trous inclus)
    - MultiPoly: conservé tel quel (trous inclus sur chaque poly)
    - GeometryCollection: union des composantes polygonales (conservant leurs trous)
    """
    if geom is None:
        return None
    g = make_valid(geom)
    if g.is_empty:
        return None

    if g.geom_type == "Polygon":
        return g  # conserve trous
    if g.geom_type == "MultiPolygon":
        return g  # conserve trous

    if isinstance(g, GeometryCollection):
        polys = []
        for sub in g.geoms:
            if sub.geom_type == "Polygon":
                polys.append(sub)
            elif sub.geom_type == "MultiPolygon":
                polys.extend(list(sub.geoms))
        if not polys:
            return None
        return unary_union(polys) if len(polys) > 1 else polys[0]

    # Autres types ignorés
    return None

def _sanitize_geom(gj: Dict[str, Any]) -> Optional[MultiPolygon]:
    """GeoJSON -> Shapely (Multi)Polygon polygonal-only, validé."""
    try:
        return _to_polygonal(shape(gj))
    except Exception:
        return None

def _build_outer_shell(geom):
    """
    Enveloppe extérieure (exterior rings uniquement) pour les tests de 'covers' du shell.
    Ici on *produit* un MultiPolygon composé des extérieurs de chaque polygone,
    sans les trous, mais on ne modifie pas la géométrie hôte originale ailleurs.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def _reproject(geom, from_epsg: str, to_epsg: str):
    """Reprojection géométrie via pyproj + shapely.ops.transform (always_xy)."""
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    return transform(transformer.transform, geom)

def _area_m2(geom_wgs84):
    """Surface en m² (en reprojetant en EPSG:2154)."""
    if geom_wgs84 is None or geom_wgs84.is_empty:
        return 0.0
    g_l93 = _reproject(geom_wgs84, "EPSG:4326", "EPSG:2154")
    return float(g_l93.area)

def _safe_carve(host_geom_wgs84, holes_union_wgs84, snap_tol_m: float = 0.02):
    """
    Soustraction robuste:
      (1) reprojection en L93
      (2) intersection des trous avec l'hôte (évite sur-comptage)
      (3) snap léger des bords (corrige micro-décalages)
      (4) difference + buffer(0) + make_valid
      (5) reprojection WGS84
    Renvoie: host_corr_wgs84, carved_effective_area_m2, delta_m2
    """
    if holes_union_wgs84 is None or holes_union_wgs84.is_empty:
        return host_geom_wgs84, 0.0, 0.0

    host_l93  = _reproject(host_geom_wgs84,  "EPSG:4326", "EPSG:2154")
    holes_l93 = _reproject(holes_union_wgs84,"EPSG:4326", "EPSG:2154")

    host_l93  = make_valid(host_l93).buffer(0)
    holes_l93 = make_valid(holes_l93).buffer(0)

    holes_in_host = host_l93.intersection(holes_l93)
    if holes_in_host.is_empty:
        host_corr_l93 = make_valid(host_l93).buffer(0)
        return _reproject(host_corr_l93, "EPSG:2154", "EPSG:4326"), 0.0, 0.0

    # Snap léger (2 cm) pour stabiliser la topologie cadastrale
    host_snapped  = snap(host_l93,  holes_in_host, snap_tol_m)
    holes_snapped = snap(holes_in_host, host_l93,  snap_tol_m)

    carved_effective_m2 = float(holes_snapped.area)
    host_corr_l93 = make_valid(host_snapped.difference(holes_snapped)).buffer(0)

    host_corr_wgs84 = _reproject(host_corr_l93, "EPSG:2154", "EPSG:4326")
    # Delta de cohérence: (aire avant - après) vs aire retranchée
    delta_m2 = abs((float(host_l93.area) - float(host_corr_l93.area)) - carved_effective_m2)
    return host_corr_wgs84, carved_effective_m2, delta_m2

# ---------- WFS helpers ----------

def _fetch_neighbors_by_bbox(minx: float, miny: float, maxx: float, maxy: float, count: int = 5000) -> List[Dict[str, Any]]:
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": LAYER_PARCELLE, "outputFormat": "application/json",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "count": int(count), "srsName": "EPSG:4326",
    }
    try:
        r = requests.get(IGN_WFS, params=params, timeout=30)
        r.raise_for_status()
        return (r.json().get("features") or [])
    except Exception:
        return []

def _inner_rings_as_polygons(geom):
    """
    Extrait les anneaux intérieurs du polygone hôte (les trous) comme Polygons.
    Utile si l'enclave est déjà encodée en trou dans la géométrie parcellaire.
    """
    rings = []
    if geom.geom_type == "Polygon":
        rings += [Polygon(r) for r in geom.interiors]
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            rings += [Polygon(r) for r in p.interiors]
    # Nettoyage / validation
    clean = []
    for r in rings:
        rr = make_valid(r)
        if not rr.is_empty and rr.area > 0:
            clean.append(rr)
    return clean

def _is_enclave_robust(gg_wgs84, host_wgs84, shell_wgs84, tol_m: float = 0.05) -> bool:
    """
    Test robuste d'enclave en L93 avec petite tolérance.
    - cas A: gg entièrement dans l'hôte (avec léger buffer) et ne touche pas son bord
    - cas B: gg couverte par le shell (exterieurs) mais pas par l'hôte (trou)
    """
    host_l93  = _reproject(host_wgs84,  "EPSG:4326", "EPSG:2154")
    shell_l93 = _reproject(shell_wgs84, "EPSG:4326", "EPSG:2154")
    gg_l93    = _reproject(gg_wgs84,    "EPSG:4326", "EPSG:2154")

    host_l93  = make_valid(host_l93).buffer(0)
    shell_l93 = make_valid(shell_l93).buffer(0)
    gg_l93    = make_valid(gg_l93).buffer(0)

    # A) à l'intérieur (avec marge) et pas tangent au bord (avec marge)
    inside_tol  = gg_l93.within(host_l93.buffer(tol_m))
    touches_tol = gg_l93.buffer(tol_m).touches(host_l93.buffer(tol_m))
    a_enclave = inside_tol and (not touches_tol)

    # B) dans l'enveloppe extérieure mais pas "couvert" par l'hôte (trou)
    b_enclave = shell_l93.covers(gg_l93) and (not host_l93.covers(gg_l93))

    return bool(a_enclave or b_enclave)

def _holes_metrics(host_geom_wgs84):
    """Retourne (holes_count, holes_area_m2, holes_geojson) si la parcelle a des trous déjà présents."""
    shell = _build_outer_shell(host_geom_wgs84)
    # travail en L93 pour des surfaces fiables
    host_l93  = _reproject(host_geom_wgs84, "EPSG:4326", "EPSG:2154")
    shell_l93 = _reproject(shell,              "EPSG:4326", "EPSG:2154")
    holes_l93 = make_valid(shell_l93.difference(host_l93)).buffer(0)
    holes_area = float(holes_l93.area)
    # reproject back for export
    holes_wgs = _reproject(holes_l93, "EPSG:2154", "EPSG:4326")
    # explode multipolygons into features
    holes_feats = []
    if not holes_wgs.is_empty:
        if holes_wgs.geom_type == "Polygon":
            holes_feats = [mapping(holes_wgs)]
        elif holes_wgs.geom_type == "MultiPolygon":
            holes_feats = [mapping(p) for p in holes_wgs.geoms if not p.is_empty]
    return len(holes_feats), holes_area, holes_feats

# ---------- Détection / soustraction des enclaves ----------

def detect_and_carve_enclaves(
    host_feature: Dict[str, Any],
    buffer_m: float = 120.0,
) -> Dict[str, Any]:
    """
    host_feature : GeoJSON "Feature" de l'IGN (une parcelle). Doit contenir 'geometry' et idéalement 'properties.code_insee'/'properties.idu'.
    buffer_m    : rayon (m) pour récupérer les parcelles voisines (pour détecter les enclaves).

    Retour:
      {
        "host_wkt_4326": "...",
        "host_corrected_wkt_4326": "...",
        "host_geojson_4326": {...},
        "host_corrected_geojson_4326": {...},
        "enclaves": [ { "area_m2": float, "geojson": {...} }, ... ],
        "stats": {
          "enclave_count": int,
          "host_area_m2": float,
          "carved_area_m2": float,
          "host_corrected_area_m2": float
        }
      }
    """
    if not host_feature or "geometry" not in host_feature:
        raise ValueError("host_feature invalide: 'geometry' manquant.")

    host_geom = _sanitize_geom(host_feature["geometry"])
    if host_geom is None:
        raise ValueError("Géométrie de parcelle invalide ou non polygonale.")

    props = host_feature.get("properties") or {}
    host_idu = props.get("idu")
    host_insee = props.get("code_insee")

    # Buffer métrique en L93
    host_l93 = _reproject(host_geom, "EPSG:4326", "EPSG:2154")
    buffer_l93 = host_l93.buffer(float(buffer_m))
    buffer_wgs = _reproject(buffer_l93, "EPSG:2154", "EPSG:4326")
    minx, miny, maxx, maxy = buffer_wgs.bounds

    # Parcelles voisines
    neighbors = _fetch_neighbors_by_bbox(minx, miny, maxx, maxy)

    shell = _build_outer_shell(host_geom)

    # 1) D'abord : récupérer les trous existants de la parcelle hôte (enclaves intégrées)
    enclaves_geoms = list(_inner_rings_as_polygons(host_geom))

    for f in neighbors:
        p = (f.get("properties") or {})
        # même commune si info dispo
        if host_insee and p.get("code_insee") and p.get("code_insee") != host_insee:
            continue
        # ignorer la parcelle hôte si on a l'IDU
        if host_idu and p.get("idu") == host_idu:
            continue

        gg = _sanitize_geom(f.get("geometry"))
        if gg is None:
            continue

        if _is_enclave_robust(gg, host_geom, shell, tol_m=0.05):
            enclaves_geoms.append(gg)

    host_area = _area_m2(host_geom)

    if enclaves_geoms:
        holes_union = unary_union(enclaves_geoms)
    else:
        holes_union = None

    if holes_union and not holes_union.is_empty:
        host_corrected, carved_effective_area_m2, delta_m2 = _safe_carve(
            host_geom, holes_union, snap_tol_m=0.02
        )
    else:
        host_corrected = host_geom
        carved_effective_area_m2 = 0.0
        delta_m2 = 0.0

    host_corr_area = _area_m2(host_corrected)

    # Métriques des trous préexistants
    holes_count, holes_area_m2, holes_geoms = _holes_metrics(host_geom)

    return {
        "host_wkt_4326": host_geom.wkt,
        "host_corrected_wkt_4326": host_corrected.wkt,
        "host_geojson_4326": mapping(host_geom),
        "host_corrected_geojson_4326": mapping(host_corrected),
        "enclaves": [
            {"area_m2": _area_m2(g), "geojson": mapping(g)}
            for g in enclaves_geoms
        ],
        "holes_in_host": {  # nouvelles infos
            "count": holes_count,
            "area_m2": holes_area_m2,
            "geojson": holes_geoms,  # liste de géométries des trous
        },
        "stats": {
            "enclave_count": len(enclaves_geoms),
            "host_area_m2": host_area,                 # aire nette (trous déjà soustraits)
            "carved_effective_area_m2": carved_effective_area_m2,  # 0 si déjà trouée
            "host_corrected_area_m2": host_corr_area,  # = host_area si déjà trouée
            "carve_consistency_delta_m2": delta_m2
        }
    }
