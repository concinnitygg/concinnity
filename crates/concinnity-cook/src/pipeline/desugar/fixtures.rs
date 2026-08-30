//! Fixtures for the desugar tests: a payload-cache hit, a morph-target `.glb`,
//! and the synthetic FBX documents the FBX passes read.

use crate::pipeline::pack::MeshCacheEntry;

// A cache map that claims a compiled payload is already in hand for the
// named asset, so every desugar pass must skip its source parse.
pub(super) fn hit_cache(name: &str) -> std::collections::HashMap<String, MeshCacheEntry> {
    let mut m = std::collections::HashMap::new();
    m.insert(
        name.to_string(),
        MeshCacheEntry {
            key: "k".to_string(),
            bytes: Some(vec![1, 2, 3]),
            capsule_scale: None,
        },
    );
    m
}

// The shared skinned fixture with one morph target ("bulge", +Y on every
// vertex) and an animation channel driving its weight from 0 to 1.
pub(super) fn morphing_skinned_glb() -> Vec<u8> {
    use crate::import::glb::test_fixtures::{f32s, make_glb, skinned_bin, skinned_json};

    let mut bin = skinned_bin(); // 136 bytes
    bin.extend(f32s(&[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])); // deltas -> 172
    bin.extend(f32s(&[0.0, 1.0])); // morph weights -> 180

    let mut json = skinned_json(true, true, true);
    json["buffers"][0]["byteLength"] = 180.into();
    let views = json["bufferViews"].as_array_mut().expect("bufferViews");
    views.push(serde_json::json!({"buffer": 0, "byteOffset": 136, "byteLength": 36}));
    views.push(serde_json::json!({"buffer": 0, "byteOffset": 172, "byteLength": 8}));
    let accessors = json["accessors"].as_array_mut().expect("accessors");
    accessors.push(
        serde_json::json!({"bufferView": 6, "componentType": 5126, "count": 3, "type": "VEC3"}),
    );
    accessors.push(serde_json::json!(
        {"bufferView": 7, "componentType": 5126, "count": 2, "type": "SCALAR"}
    ));
    json["meshes"][0]["primitives"][0]["targets"] = serde_json::json!([{"POSITION": 6}]);
    json["meshes"][0]["extras"] = serde_json::json!({"targetNames": ["bulge"]});
    json["animations"][0]["samplers"]
        .as_array_mut()
        .expect("samplers")
        .push(serde_json::json!({"input": 4, "output": 7, "interpolation": "LINEAR"}));
    json["animations"][0]["channels"]
        .as_array_mut()
        .expect("channels")
        .push(serde_json::json!({"sampler": 2, "target": {"node": 0, "path": "weights"}}));

    make_glb(&json, Some(&bin))
}

// Synthetic binary FBX containers. The importer walks a node tree, so the
// fixtures are described as one and serialized by `write_fbx`; no binary
// asset needs to live in the repo.
pub(super) enum Attr {
    Int(i64),
    Double(f64),
    Text(String),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    Longs(Vec<i64>),
    Floats(Vec<f32>),
}

pub(super) struct Node {
    name: &'static str,
    attrs: Vec<Attr>,
    children: Vec<Node>,
}

pub(super) fn node(name: &'static str, attrs: Vec<Attr>, children: Vec<Node>) -> Node {
    Node {
        name,
        attrs,
        children,
    }
}

// An FBX object's second attribute: the authored name, the object class,
// and the `\0\u{1}` separator between them.
pub(super) fn object_name(name: &str, class: &str) -> Attr {
    Attr::Text(format!("{name}\u{0}\u{1}{class}"))
}

pub(super) fn connection(child: i64, parent: i64) -> Node {
    node(
        "C",
        vec![
            Attr::Text("OO".to_string()),
            Attr::Int(child),
            Attr::Int(parent),
        ],
        Vec::new(),
    )
}

// An object-to-property connection: the parent's named property is what
// the child drives.
pub(super) fn property_connection(child: i64, parent: i64, property: &str) -> Node {
    node(
        "C",
        vec![
            Attr::Text("OP".to_string()),
            Attr::Int(child),
            Attr::Int(parent),
            Attr::Text(property.to_string()),
        ],
        Vec::new(),
    )
}

pub(super) fn write_fbx(nodes: &[Node]) -> Vec<u8> {
    use fbxcel::low::FbxVersion;
    use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

    fn emit<W: std::io::Write + std::io::Seek>(w: &mut Writer<W>, n: &Node) -> std::io::Result<()> {
        {
            let mut attrs = w.new_node(n.name).expect("open node");
            for a in &n.attrs {
                match a {
                    Attr::Int(v) => attrs.append_i64(*v),
                    Attr::Double(v) => attrs.append_f64(*v),
                    Attr::Text(s) => attrs.append_string_direct(s),
                    Attr::Doubles(v) => attrs.append_arr_f64_from_iter(None, v.iter().copied()),
                    Attr::Ints(v) => attrs.append_arr_i32_from_iter(None, v.iter().copied()),
                    Attr::Longs(v) => attrs.append_arr_i64_from_iter(None, v.iter().copied()),
                    Attr::Floats(v) => attrs.append_arr_f32_from_iter(None, v.iter().copied()),
                }
                .expect("append attribute");
            }
        }
        for c in &n.children {
            emit(w, c)?;
        }
        w.close_node().expect("close node");
        Ok(())
    }

    let mut w =
        Writer::new(std::io::Cursor::new(Vec::new()), FbxVersion::V7_4).expect("fbx writer");
    for n in nodes {
        emit(&mut w, n).expect("emit node");
    }
    w.finalize_and_flush(&FbxFooter::default())
        .expect("finalize")
        .into_inner()
}

// One triangle as an FBX Geometry object. The last corner of a polygon is
// stored bitwise-negated, which is how the importer finds polygon bounds.
pub(super) fn triangle_geometry(id: i64) -> Node {
    node(
        "Geometry",
        vec![
            Attr::Int(id),
            object_name("tri", "Geometry"),
            Attr::Text("Mesh".to_string()),
        ],
        vec![
            node(
                "Vertices",
                vec![Attr::Doubles(vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                ])],
                Vec::new(),
            ),
            node(
                "PolygonVertexIndex",
                vec![Attr::Ints(vec![0, 1, !2])],
                Vec::new(),
            ),
        ],
    )
}

// One Model connected to one Geometry: `parse_fbx` yields a single
// primitive from it.
pub(super) fn static_triangle_fbx() -> Vec<u8> {
    const GEOMETRY: i64 = 1000;
    const MODEL: i64 = 2000;
    write_fbx(&[
        node(
            "Objects",
            Vec::new(),
            vec![
                triangle_geometry(GEOMETRY),
                node(
                    "Model",
                    vec![
                        Attr::Int(MODEL),
                        object_name("tri", "Model"),
                        Attr::Text("Mesh".to_string()),
                    ],
                    Vec::new(),
                ),
            ],
        ),
        node("Connections", Vec::new(), vec![connection(GEOMETRY, MODEL)]),
    ])
}

// FBX time unit: ticks per second.
pub(super) const KTIME_PER_SEC: i64 = 46_186_158_000;

pub(super) fn skinned_triangle_fbx() -> Vec<u8> {
    skinned_fbx(false)
}

// The same triangle bound to a one-bone skin: a Skin deformer over the
// geometry, a Cluster linking every control point to the bone Model at an
// identity bind, and a unit scale of 100 so file units are already meters.
// With `animated`, a one-second stack slides the bone 0 -> 2 along X.
pub(super) fn skinned_fbx(animated: bool) -> Vec<u8> {
    const GEOMETRY: i64 = 3000;
    const MESH_MODEL: i64 = 4000;
    const BONE: i64 = 5000;
    const SKIN: i64 = 6000;
    const CLUSTER: i64 = 7000;
    const STACK: i64 = 8000;
    const LAYER: i64 = 8100;
    const CURVE_NODE: i64 = 8200;
    const CURVE: i64 = 8300;
    let identity = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let unit_scale = node(
        "GlobalSettings",
        Vec::new(),
        vec![node(
            "Properties70",
            Vec::new(),
            vec![node(
                "P",
                vec![
                    Attr::Text("UnitScaleFactor".to_string()),
                    Attr::Text("double".to_string()),
                    Attr::Text("Number".to_string()),
                    Attr::Text(String::new()),
                    Attr::Double(100.0),
                ],
                Vec::new(),
            )],
        )],
    );
    let mut objects = vec![
        triangle_geometry(GEOMETRY),
        node(
            "Model",
            vec![
                Attr::Int(MESH_MODEL),
                object_name("mesh", "Model"),
                Attr::Text("Mesh".to_string()),
            ],
            Vec::new(),
        ),
        node(
            "Model",
            vec![
                Attr::Int(BONE),
                object_name("Root", "Model"),
                Attr::Text("LimbNode".to_string()),
            ],
            Vec::new(),
        ),
        node(
            "Deformer",
            vec![
                Attr::Int(SKIN),
                object_name("skin", "Deformer"),
                Attr::Text("Skin".to_string()),
            ],
            Vec::new(),
        ),
        node(
            "Deformer",
            vec![
                Attr::Int(CLUSTER),
                object_name("cluster", "SubDeformer"),
                Attr::Text("Cluster".to_string()),
            ],
            vec![
                node("Indexes", vec![Attr::Ints(vec![0, 1, 2])], Vec::new()),
                node(
                    "Weights",
                    vec![Attr::Doubles(vec![1.0, 1.0, 1.0])],
                    Vec::new(),
                ),
                node(
                    "TransformLink",
                    vec![Attr::Doubles(identity.clone())],
                    Vec::new(),
                ),
                node("Transform", vec![Attr::Doubles(identity)], Vec::new()),
            ],
        ),
    ];
    let mut connections = vec![
        connection(GEOMETRY, MESH_MODEL),
        connection(SKIN, GEOMETRY),
        connection(CLUSTER, SKIN),
        connection(BONE, CLUSTER),
    ];

    if animated {
        objects.extend([
            node(
                "AnimationStack",
                vec![Attr::Int(STACK), object_name("wave", "AnimStack")],
                Vec::new(),
            ),
            node(
                "AnimationLayer",
                vec![Attr::Int(LAYER), object_name("Base Layer", "AnimLayer")],
                Vec::new(),
            ),
            node(
                "AnimationCurveNode",
                vec![Attr::Int(CURVE_NODE), object_name("T", "AnimCurveNode")],
                Vec::new(),
            ),
            node(
                "AnimationCurve",
                vec![Attr::Int(CURVE), object_name("", "AnimCurve")],
                vec![
                    node(
                        "KeyTime",
                        vec![Attr::Longs(vec![0, KTIME_PER_SEC])],
                        Vec::new(),
                    ),
                    node(
                        "KeyValueFloat",
                        vec![Attr::Floats(vec![0.0, 2.0])],
                        Vec::new(),
                    ),
                ],
            ),
        ]);
        connections.extend([
            connection(LAYER, STACK),
            connection(CURVE_NODE, LAYER),
            property_connection(CURVE, CURVE_NODE, "d|X"),
            property_connection(CURVE_NODE, BONE, "Lcl Translation"),
        ]);
    }

    write_fbx(&[
        unit_scale,
        node("Objects", Vec::new(), objects),
        node("Connections", Vec::new(), connections),
    ])
}
