//! The semantic id table: Onshape deterministic ids -> references that are
//! meaningful in the interpreter's own model and stable across
//! configurations.
//!
//! JSON form (one object keyed by id):
//!
//! ```json
//! {
//!   "JDC": {"kind": "plane", "origin": [0,0,0], "normal": [0,0,1], "x": [1,0,0]},
//!   "JHD": {"kind": "body", "feature": "FqNDP2FKjRTvKVH_0"},
//!   "JGC": {"kind": "region", "sketch": "FwPVGcv7S9TxyD7_0", "ids": ["a.bottom", "a.left", "a.right", "a.top"]},
//!   "JHG": {"kind": "face", "feature": "FqNDP2FKjRTvKVH_0", "role": "cap"},
//!   "RFB9": {"kind": "edge", "side": "top",
//!            "a": {"feature": "F...", "role": "cap"}, "b": {"feature": "F...", "role": "wall"},
//!            "hint": [0.1, 0.2, 0.05]},
//!   "I1":  {"kind": "sketch_point", "sketch": "F...", "id": "abc.start"},
//!   "RYCB": {"kind": "sketch_line", "sketch": "F...", "id": "abc"}
//! }
//! ```
//!
//! Entries may also carry `"note"` (free text) and `"source"` (how the
//! entry was obtained: `"derived"`, `"api"`, `"manual"`).
//!
//! Deterministic ids are not stable across configurations in this model
//! (a block shift moves every later id when a feature's output count
//! changes), so references are also stored per **query slot**
//! (`"<feature id>/<parameter id>/<query index>/<id index>"`) under
//! `"slots"`, and a fillet's or chamfer's edges per feature under
//! `"edges"` as the face pairs its blend faces join. Lookups try the slot,
//! then the id. The top-level JSON is `{"ids": {...}, "slots": {...},
//! "edges": {...}}`; a bare id map is accepted for older files.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaceRole {
    /// The face at the far end of an extrude (or the flat top / bottom a
    /// feature made on the solid).
    Cap,
    /// The face at the sketch plane end of an extrude.
    Start,
    /// The vertical side faces an extrude made.
    Wall,
    /// A loft's side surface.
    Loft,
    /// A fillet / chamfer face.
    Blend,
    /// The solid's footprint side wall (any feature).
    Outline,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionEntry {
    pub sketch: String,
    pub ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pick: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceRef {
    pub feature: String,
    pub role: FaceRole,
    /// For a loft's side surface: the box of the one panel meant (model
    /// frame, metres), when the loft has several.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 6]>,
    /// Points on the face (model frame, metres): the blend's rails that
    /// lie on it. They name the loft panel(s) the face is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pts: Option<Vec<[f64; 3]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideRef {
    Top,
    Bottom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ref {
    Plane {
        origin: [f64; 3],
        normal: [f64; 3],
        x: [f64; 3],
        /// A datum plane (standard or cPlane) rather than a body face; a
        /// sketch on a body face is split by the face's edges.
        #[serde(default)]
        datum: bool,
    },
    Body {
        feature: String,
    },
    Region {
        sketch: String,
        ids: Vec<String>,
        /// Which region when several share the same boundary ids (two
        /// lobes of one lens): index in centroid order (u, then v).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pick: Option<usize>,
    },
    Face {
        feature: String,
        role: FaceRole,
    },
    Edge {
        a: FaceRef,
        b: FaceRef,
        side: SideRef,
        #[serde(default)]
        hint: Option<[f64; 3]>,
    },
    /// All regions a parameter (or one loft profile item) names, as a set:
    /// ids regroup between configurations and a region may not exist at
    /// every one, so the list is matched as a whole.
    RegionList {
        regions: Vec<RegionEntry>,
    },
    /// A vertical wall face an extrude made along one boundary entity of its
    /// region, used as a loft profile; `t0..t1` is the fraction of that
    /// entity the face covered at the configuration it was resolved at.
    WallFace {
        extrude: String,
        sketch: String,
        ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pick: Option<usize>,
        entity: String,
        t0: f64,
        t1: f64,
        /// The face's two ends on the entity (world, metres, at the sketch
        /// plane): more robust than the fractions when the entity is split
        /// into several pieces.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ends: Option<[[f64; 3]; 2]>,
    },
    /// Per sketch (key `<feature>/sketch`): whether the seed fit's frame is
    /// mirrored relative to Onshape's sketch normal, decided against the API
    /// resolutions (which way the arcs turn).
    SketchFrame {
        flipped: bool,
    },
    /// A loft guide made by a compositeCurve feature (its `edges` slots are
    /// the sketch curves to chain).
    Guide {
        composite: String,
    },
    SketchPoint {
        sketch: String,
        id: String,
    },
    SketchLine {
        sketch: String,
        id: String,
    },
    /// Known to be unresolvable for now, with the reason.
    Unresolved {
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    #[serde(flatten)]
    pub r: Ref,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// One edge of a fillet / chamfer as the pair of faces its blend joins,
/// with the blend face's bounding box (model frame, metres) at the
/// configuration it was resolved at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeRef {
    pub a: FaceRef,
    pub b: FaceRef,
    pub side: SideRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint_box: Option<[f64; 6]>,
    /// The blend face's rails (its tangent edges on the two faces), each
    /// as start / middle / end points (model frame, metres): the edge
    /// runs between them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rails: Option<Vec<[[f64; 3]; 3]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdTable {
    #[serde(default, rename = "ids")]
    pub entries: HashMap<String, Entry>,
    #[serde(default)]
    pub slots: HashMap<String, Entry>,
    #[serde(default)]
    pub edges: HashMap<String, Vec<EdgeRef>>,
    /// Features suppressed at the configuration the table was built at
    /// (their slots are absent, not unresolved).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suppressed: Vec<String>,
}

fn sorted_map<T: Serialize>(m: &HashMap<String, T>) -> serde_json::Value {
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort();
    let mut map = serde_json::Map::new();
    for k in keys {
        map.insert(k.clone(), serde_json::to_value(&m[k]).unwrap());
    }
    serde_json::Value::Object(map)
}

impl IdTable {
    pub fn from_json(text: &str) -> Result<IdTable, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if v.get("ids").is_some() || v.get("slots").is_some() || v.get("edges").is_some() {
            serde_json::from_value(v).map_err(|e| e.to_string())
        } else {
            let entries: HashMap<String, Entry> = serde_json::from_value(v).map_err(|e| e.to_string())?;
            Ok(IdTable { entries, ..Default::default() })
        }
    }

    pub fn to_json(&self) -> String {
        let mut map = serde_json::Map::new();
        map.insert("ids".into(), sorted_map(&self.entries));
        map.insert("slots".into(), sorted_map(&self.slots));
        map.insert("edges".into(), sorted_map(&self.edges));
        if !self.suppressed.is_empty() {
            map.insert("suppressed".into(), serde_json::to_value(&self.suppressed).unwrap());
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap()
    }

    /// The slot key for the `ii`-th id of the `qi`-th query of a parameter.
    pub fn slot_key(feature: &str, param: &str, qi: usize, ii: usize) -> String {
        format!("{feature}/{param}/{qi}/{ii}")
    }

    /// The key of a parameter's whole region list (`item` for one loft
    /// profile item).
    pub fn list_key(feature: &str, param: &str, item: Option<usize>) -> String {
        match item {
            Some(i) => format!("{feature}/{param}/item{i}"),
            None => format!("{feature}/{param}"),
        }
    }

    pub fn insert_slot(&mut self, key: &str, r: Ref, source: &str) {
        self.slots.insert(key.to_string(), Entry { r, note: None, source: Some(source.to_string()) });
    }

    /// Slot entry first, then the id.
    pub fn lookup(&self, slot: &str, id: &str) -> Option<&Ref> {
        match self.slots.get(slot).map(|e| &e.r) {
            Some(Ref::Unresolved { .. }) | None => self.get(id),
            some => some,
        }
    }

    pub fn get(&self, id: &str) -> Option<&Ref> {
        self.entries.get(id).map(|e| &e.r)
    }

    pub fn insert(&mut self, id: &str, r: Ref, source: &str) {
        self.entries.insert(id.to_string(), Entry { r, note: None, source: Some(source.to_string()) });
    }

    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Whether the id has a usable entry (not merely marked unresolved).
    pub fn resolved(&self, id: &str) -> bool {
        !matches!(self.get(id), None | Some(Ref::Unresolved { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip() {
        let mut t = IdTable::default();
        t.insert("JDC", Ref::Plane { origin: [0.0; 3], normal: [0.0, 0.0, 1.0], x: [1.0, 0.0, 0.0], datum: true }, "derived");
        t.insert(
            "RFB9",
            Ref::Edge {
                a: FaceRef { feature: "F1".into(), role: FaceRole::Cap, bbox: None, pts: None },
                b: FaceRef { feature: "F2".into(), role: FaceRole::Wall, bbox: None, pts: None },
                side: SideRef::Top,
                hint: None,
            },
            "api",
        );
        let text = t.to_json();
        let back = IdTable::from_json(&text).unwrap();
        assert_eq!(back.get("JDC"), t.get("JDC"));
        assert_eq!(back.get("RFB9"), t.get("RFB9"));
        assert!(text.contains("\"kind\": \"edge\""));
    }
}
