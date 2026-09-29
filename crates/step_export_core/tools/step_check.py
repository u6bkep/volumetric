"""Check STEP files in an independent kernel (OpenCascade, through FreeCAD).

    FreeCADCmd step_check.py --pass <file.step> [...]
    STEP_CHECK_REF=<reference.step> FreeCADCmd step_check.py --pass <file.step>

For every file: whether each solid is valid and closed by the kernel's own
checker, its volume, area, bounding box and a census of its face types, as
one JSON object per line. With STEP_CHECK_REF the first solid is also
compared with the reference file's first solid: the difference of their
volumes, and the largest distance from points spread over each one's faces
to the other. (No boolean of the two: on solids that share faces, which a
file and its reference mostly do, OpenCascade 7.9 returns wrong volumes
without complaint.)

FreeCAD takes every argument for a file to open, so arguments after
`--pass` are the way in. Lengths are millimetres whatever the file's unit.
"""

import json
import os
import sys

import FreeCAD  # noqa: F401  (must be loaded before Part)
import Part


def solids_of(path):
    shape = Part.Shape()
    shape.read(path)
    return shape, list(shape.Solids)


def census(solid):
    kinds = {}
    for face in solid.Faces:
        name = type(face.Surface).__name__
        kinds[name] = kinds.get(name, 0) + 1
    return dict(sorted(kinds.items()))


def describe(path):
    shape, solids = solids_of(path)
    out = {"file": path, "solids": [], "loose_faces": len(shape.Faces) - sum(len(s.Faces) for s in solids)}
    for solid in solids:
        bb = solid.BoundBox
        entry = {
            "valid": bool(solid.isValid()),
            "closed": all(shell.isClosed() for shell in solid.Shells),
            "shells": len(solid.Shells),
            "faces": len(solid.Faces),
            "edges": len(solid.Edges),
            "vertices": len(solid.Vertexes),
            "volume_mm3": solid.Volume,
            "area_mm2": solid.Area,
            "bbox_mm": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
            "kinds": census(solid),
        }
        if not entry["valid"]:
            try:
                solid.check(True)
            except Exception as err:  # the checker reports by raising
                entry["check"] = str(err)[:2000]
        out["solids"].append(entry)
    return out, solids


def face_points(solid, per_face):
    """Points spread over the faces, through the kernel's own mesher."""
    pts = []
    for face in solid.Faces:
        verts, _ = face.tessellate(max(face.BoundBox.DiagonalLength / per_face, 0.05))
        pts.extend(verts)
    return pts


def one_way(points, other):
    shell = other.Shells[0]
    worst = 0.0
    where = None
    for p in points:
        d = shell.distToShape(Part.Vertex(p))[0]
        if d > worst:
            worst, where = d, [p.x, p.y, p.z]
    return worst, where


def compare(solid, ref, per_face):
    out = {"volume_file_minus_reference_mm3": solid.Volume - ref.Volume}
    d, p = one_way(face_points(solid, per_face), ref)
    out["file_to_reference_mm"] = {"max": d, "at": p}
    d, p = one_way(face_points(ref, per_face), solid)
    out["reference_to_file_mm"] = {"max": d, "at": p}
    return out


def main():
    args = sys.argv[sys.argv.index("--pass") + 1 :] if "--pass" in sys.argv else []
    if not args:
        print(__doc__)
        return 2
    ref_path = os.environ.get("STEP_CHECK_REF")
    per_face = int(os.environ.get("STEP_CHECK_POINTS", "12"))
    ref = solids_of(ref_path)[1][0] if ref_path else None
    bad = 0
    for path in args:
        report, solids = describe(path)
        if not solids or report["loose_faces"] or any(not (s["valid"] and s["closed"]) for s in report["solids"]):
            bad += 1
        if ref is not None and solids:
            report["against"] = ref_path
            report["difference"] = compare(solids[0], ref, per_face)
        print("STEP_CHECK " + json.dumps(report))
    return 1 if bad else 0


code = main()
sys.stdout.flush()
os._exit(code)
