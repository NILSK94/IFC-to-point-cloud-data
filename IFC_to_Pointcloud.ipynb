import ifcopenshell
import ifcopenshell.geom
import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path
import math, re
import trimesh

# ========= Parameter =========
ifc_path   = XXXX
output_dir = XXXX

spacing_m  = 0.02            # Ziel-Punktabstand (kleiner = dichter)
max_pts_per_elem = 200_000   # Safety-Cap pro Objekt, verhindert Millionenpunkte
save_per_object_ply = True   # Pro Objekt zusätzlich PLY speichern
remove_inside_points = True  # Punkte entfernen, die im Inneren anderer Objekte liegen
near_surface_eps = 0.003     # Punkte entfernen, die näher als eps an fremden Oberflächen liegen (m); 0 zum Deaktivieren

# ========= Utils =========
def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))
    return s[:80]

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def sample_triangle(a, b, c, n):
    """Gleichmäßiges Flächensampling via baryzentrische Koordinaten (sqrt-Trick)."""
    if n <= 0: 
        return np.empty((0,3), float)
    u = np.random.rand(n); v = np.random.rand(n); su = np.sqrt(u)
    return (1 - su)[:,None]*a + (su*(1-v))[:,None]*b + (su*v)[:,None]*c

def sample_mesh_dense(vertices, faces, spacing, cap=None):
    """Erzeugt eine dichte Punktwolke über alle Dreiecke eines Meshes."""
    V = np.asarray(vertices, float).reshape(-1,3)
    F = np.asarray(faces, np.int32).reshape(-1,3)
    if len(V)==0 or len(F)==0:
        return np.empty((0,3), float)

    # Erwartete Punktzahl schätzen (für Cap)
    est_total = 0
    for tri in F:
        a, b, c = V[tri]
        A = triangle_area(a, b, c)
        est_total += max(1, int(math.ceil(A / (spacing*spacing))))
    if cap and est_total > cap:
        spacing *= math.sqrt(est_total / cap)  # Dichte adaptiv reduzieren

    outs = []
    for tri in F:
        a, b, c = V[tri]
        A = triangle_area(a, b, c)
        n = max(1, int(math.ceil(A / (spacing*spacing))))
        outs.append(sample_triangle(a, b, c, n))
    return np.vstack(outs) if outs else np.empty((0,3), float)

# ========= IFC öffnen & Settings robust setzen =========
ifc = ifcopenshell.open(ifc_path)
settings = ifcopenshell.geom.settings()

def set_flag(s, name, value):
    const = getattr(s, name, None)
    if const is not None:
        s.set(const, value)

set_flag(settings, "USE_WORLD_COORDS", True)
set_flag(settings, "CONVERT_BACK_UNITS", True)
set_flag(settings, "DISABLE_OPENING_SUBTRACTIONS", False)
set_flag(settings, "APPLY_DEFAULT_STYLE", False)

# ========= Ausgabeordner =========
out_dir = Path(output_dir)
(out_dir / "per_object").mkdir(parents=True, exist_ok=True)

# ========= Schritt 1: Geometrien & dichte Punkte erzeugen =========
elems = []  # [{label,gid,verts,faces,points,...}]
for elem in ifc.by_type("IfcProduct"):
    # In der Praxis haben v.a. IfcElement-Instanzen Geometrie
    if not elem.is_a("IfcElement"):
        continue
    try:
        shape = ifcopenshell.geom.create_shape(settings, elem)
    except Exception:
        continue

    verts = np.array(shape.geometry.verts, float).reshape(-1, 3)
    faces = np.array(shape.geometry.faces, np.int32).reshape(-1, 3)
    if verts.size == 0 or faces.size == 0:
        continue

    dense = sample_mesh_dense(verts, faces, spacing=spacing_m, cap=max_pts_per_elem)
    if dense.size == 0:
        continue

    elems.append({
        "label": elem.is_a(),
        "gid": getattr(elem, "GlobalId", "NoGID"),
        "verts": verts,
        "faces": faces,
        "points": dense
    })

if not elems:
    print("Keine verwertbaren Geometrien gefunden.")
    raise SystemExit

# ========= Schritt 2: Trimesh-Objekte + AABBs bauen =========
for e in elems:
    e["tmesh"] = trimesh.Trimesh(vertices=e["verts"], faces=e["faces"], process=False)
    e["aabb_min"] = e["verts"].min(axis=0)
    e["aabb_max"] = e["verts"].max(axis=0)

# ========= Schritt 3: Überschneidungen / verdeckte Punkte filtern =========
filtered_pts_all = []
filtered_lbl_all = []
index_rows = []

for idx_i, ei in enumerate(elems, start=1):
    pts = ei["points"]
    keep_mask = np.ones(len(pts), dtype=bool)

    # Kandidaten (alle anderen Objekte)
    others = [ej for ej in elems if ej is not ei]

    # In Batches arbeiten (Memory/Speed)
    batch = 50000
    for start in range(0, len(pts), batch):
        P = pts[start:start+batch]

        # Initiale Masken
        inside_remove = np.zeros(len(P), dtype=bool)
        near_remove   = np.zeros(len(P), dtype=bool)

        # Gegen andere Objekte testen
        for ej in others:
            # AABB-Gate: nur Punkte prüfen, die in der AABB des anderen Meshes liegen
            in_aabb = np.all((P >= ej["aabb_min"]) & (P <= ej["aabb_max"]), axis=1)
            if not np.any(in_aabb):
                continue

            P_in = P[in_aabb]

            # 3.1: Inside-Test (Punkte im Inneren anderer Meshes entfernen)
            if remove_inside_points:
                try:
                    inside = ej["tmesh"].contains(P_in)  # bool array (M,)
                except Exception:
                    inside = np.zeros(len(P_in), dtype=bool)
                inside_remove[in_aabb] |= inside

            # 3.2: Nah-an-Oberfläche-Test (optional)
            if near_surface_eps and near_surface_eps > 0:
                try:
                    # Gibt (M,) Distanzen zur Oberfläche zurück
                    dists = ej["tmesh"].nearest.distance(P_in)
                except Exception:
                    dists = np.full(len(P_in), np.inf)
                near = dists < near_surface_eps
                near_remove[in_aabb] |= near

        # Punkte entfernen, die inside ODER near sind
        rem = inside_remove | near_remove
        keep_mask[start:start+batch] &= ~rem

    kept = pts[keep_mask]
    filtered_pts_all.append(kept)
    filtered_lbl_all.extend([ei["label"]] * len(kept))

    # Pro-Objekt speichern (gefiltert)
    fname = f"{idx_i:04d}_{safe_name(ei['label'])}_{safe_name(ei['gid'])}"
    df_obj = pd.DataFrame(kept, columns=["x","y","z"])
    df_obj["label"] = ei["label"]
    (out_dir / "per_object").mkdir(parents=True, exist_ok=True)
    df_obj.to_csv((out_dir / "per_object" / f"{fname}.csv").as_posix(), index=False)

    if save_per_object_ply:
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(kept)
        o3d.io.write_point_cloud((out_dir / "per_object" / f"{fname}.ply").as_posix(), pcd_obj, write_ascii=True)

    index_rows.append({
        "index": idx_i,
        "global_id": ei["gid"],
        "label": ei["label"],
        "points_before": len(ei["points"]),
        "points_after": len(kept),
        "csv": (out_dir / "per_object" / f"{fname}.csv").as_posix()
    })

# ========= Gesamt speichern =========
pts_all = np.vstack(filtered_pts_all) if filtered_pts_all else np.empty((0,3))
df_all = pd.DataFrame(pts_all, columns=["x","y","z"])
df_all["label"] = filtered_lbl_all
df_all.to_csv((out_dir / "all_points_visible_only.csv").as_posix(), index=False)

pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(pts_all)
o3d.io.write_point_cloud((out_dir / "all_points_visible_only.ply").as_posix(), pcd_all, write_ascii=True)

pd.DataFrame(index_rows).to_csv((out_dir / "objects_index.csv").as_posix(), index=False)

print(f"✓ Gesamt CSV: {out_dir/'all_points_visible_only.csv'}")
print(f"✓ Gesamt PLY: {out_dir/'all_points_visible_only.ply'}")
print(f"✓ Pro-Objekt: {out_dir/'per_object'}")
print(f"✓ Index:      {out_dir/'objects_index.csv'}")

# ========= Optional: Visualisierung (Subsample, damit es nicht laggt) =========
try:
    view_fraction = 0.2  # 20% zeigen; auf 1.0 setzen für alles
    vis_pcd = pcd_all
    if view_fraction < 1.0 and len(pts_all) > 0:
        idx = np.random.choice(len(pts_all), size=max(1, int(len(pts_all) * view_fraction)), replace=False)
        vis_pcd = pcd_all.select_by_index(idx.tolist())
    o3d.visualization.draw_geometries([vis_pcd])
except Exception as e:
    print("Visualisierung übersprungen:", e)
