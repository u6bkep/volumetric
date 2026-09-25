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

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct FaceRef {
    pub feature: String,
    pub role: FaceRole,
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
    },
    Body {
        feature: String,
    },
    Region {
        sketch: String,
        ids: Vec<String>,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IdTable {
    pub entries: HashMap<String, Entry>,
}

impl IdTable {
    pub fn from_json(text: &str) -> Result<IdTable, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }

    pub fn to_json(&self) -> String {
        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();
        let mut map = serde_json::Map::new();
        for k in keys {
            map.insert(k.clone(), serde_json::to_value(&self.entries[k]).unwrap());
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap()
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
        t.insert("JDC", Ref::Plane { origin: [0.0; 3], normal: [0.0, 0.0, 1.0], x: [1.0, 0.0, 0.0] }, "derived");
        t.insert(
            "RFB9",
            Ref::Edge {
                a: FaceRef { feature: "F1".into(), role: FaceRole::Cap },
                b: FaceRef { feature: "F2".into(), role: FaceRole::Wall },
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
