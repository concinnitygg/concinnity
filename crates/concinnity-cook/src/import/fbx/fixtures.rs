// Synthetic binary FBX documents for the importer tests. A node tree is
// described in memory and written out with the fbxcel binary writer, so each
// test declares only the scene shape it needs.

use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use fbxcel::low::FbxVersion;
use fbxcel::tree::v7400::Tree;
use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

pub(super) enum Attr {
    I32(i32),
    I64(i64),
    F64(f64),
    Str(String),
    ArrI32(Vec<i32>),
    ArrI64(Vec<i64>),
    ArrF32(Vec<f32>),
    ArrF64(Vec<f64>),
}

pub(super) struct Node {
    name: String,
    attrs: Vec<Attr>,
    children: Vec<Node>,
}

pub(super) fn node(name: &str) -> Node {
    Node {
        name: name.to_string(),
        attrs: Vec::new(),
        children: Vec::new(),
    }
}

impl Node {
    fn attr(mut self, a: Attr) -> Self {
        self.attrs.push(a);
        self
    }

    pub(super) fn int32(self, v: i32) -> Self {
        self.attr(Attr::I32(v))
    }

    pub(super) fn int64(self, v: i64) -> Self {
        self.attr(Attr::I64(v))
    }

    pub(super) fn float(self, v: f64) -> Self {
        self.attr(Attr::F64(v))
    }

    pub(super) fn text(self, v: &str) -> Self {
        self.attr(Attr::Str(v.to_string()))
    }

    pub(super) fn arr_i32(self, v: Vec<i32>) -> Self {
        self.attr(Attr::ArrI32(v))
    }

    pub(super) fn arr_i64(self, v: Vec<i64>) -> Self {
        self.attr(Attr::ArrI64(v))
    }

    pub(super) fn arr_f32(self, v: Vec<f32>) -> Self {
        self.attr(Attr::ArrF32(v))
    }

    pub(super) fn arr_f64(self, v: Vec<f64>) -> Self {
        self.attr(Attr::ArrF64(v))
    }

    pub(super) fn child(mut self, c: Node) -> Self {
        self.children.push(c);
        self
    }

    pub(super) fn children(mut self, cs: Vec<Node>) -> Self {
        self.children.extend(cs);
        self
    }
}

fn emit<W: Write + Seek>(w: &mut Writer<W>, n: &Node) {
    {
        let mut a = w.new_node(&n.name).expect("open node");
        for attr in &n.attrs {
            match attr {
                Attr::I32(v) => a.append_i32(*v),
                Attr::I64(v) => a.append_i64(*v),
                Attr::F64(v) => a.append_f64(*v),
                Attr::Str(s) => a.append_string_direct(s),
                Attr::ArrI32(v) => a.append_arr_i32_from_iter(None, v.iter().copied()),
                Attr::ArrI64(v) => a.append_arr_i64_from_iter(None, v.iter().copied()),
                Attr::ArrF32(v) => a.append_arr_f32_from_iter(None, v.iter().copied()),
                Attr::ArrF64(v) => a.append_arr_f64_from_iter(None, v.iter().copied()),
            }
            .expect("write attribute");
        }
    }
    for c in &n.children {
        emit(w, c);
    }
    w.close_node().expect("close node");
}

// A written .fbx plus the temporary directory holding it; texture paths are
// resolved relative to that directory, so tests keep it alive.
pub(crate) struct FbxFile {
    dir: tempfile::TempDir,
    path: String,
}

impl FbxFile {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    pub(super) fn dir(&self) -> &Path {
        self.dir.path()
    }
}

pub(super) fn write(roots: Vec<Node>) -> FbxFile {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fixture.fbx");
    let sink = std::fs::File::create(&path).expect("create fbx");
    let mut w = Writer::new(BufWriter::new(sink), FbxVersion::V7_4).expect("fbx writer");
    for n in &roots {
        emit(&mut w, n);
    }
    w.finalize_and_flush(&FbxFooter::default())
        .expect("finalize fbx");
    FbxFile {
        path: path.to_string_lossy().into_owned(),
        dir,
    }
}

// Write and immediately reload; the tree owns its data, so the file is not
// needed afterwards.
pub(super) fn tree(roots: Vec<Node>) -> Tree {
    let file = write(roots);
    super::load_tree(file.path()).expect("load tree")
}

// The forward-slashed path `resolve_texture_path` produces for `rel` under
// `dir`, so expectations do not bake in the host separator.
pub(super) fn under(dir: &Path, rel: &str) -> String {
    dir.join(rel).to_string_lossy().replace('\\', "/")
}

pub(super) fn assert_vec3_eq(a: [f32; 3], b: [f32; 3]) {
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < 1e-4, "component {i}: {a:?} vs {b:?}");
    }
}

struct WarnCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        *meta.level() == tracing::Level::WARN
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

// Runs `f` and reports how many warn-level events it emitted. A live
// subscriber is required: `tracing` skips argument evaluation entirely when
// nothing is listening, so the warning paths never run without one.
pub(super) fn count_warnings<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let out = tracing::subscriber::with_default(WarnCounter(count.clone()), f);
    (out, count.load(std::sync::atomic::Ordering::Relaxed))
}

// FBX object header: id, "Name\0\u{1}Class", sub-class.
pub(super) fn object(kind: &str, id: i64, name: &str, sub_class: &str) -> Node {
    node(kind)
        .int64(id)
        .text(&format!("{name}\u{0}\u{1}{kind}"))
        .text(sub_class)
}

pub(super) fn properties70(props: Vec<Node>) -> Node {
    node("Properties70").children(props)
}

pub(super) fn p_vec3(name: &str, v: [f64; 3]) -> Node {
    node("P")
        .text(name)
        .text("Vector3D")
        .text("Vector")
        .text("A")
        .float(v[0])
        .float(v[1])
        .float(v[2])
}

pub(super) fn p_scalar(name: &str, v: f64) -> Node {
    node("P")
        .text(name)
        .text("double")
        .text("Number")
        .text("A")
        .float(v)
}

pub(super) fn p_time(name: &str, ticks: i64) -> Node {
    node("P")
        .text(name)
        .text("KTime")
        .text("Time")
        .text("")
        .int64(ticks)
}

pub(super) fn objects(children: Vec<Node>) -> Node {
    node("Objects").children(children)
}

pub(super) fn connections(children: Vec<Node>) -> Node {
    node("Connections").children(children)
}

pub(super) fn oo(child: i64, parent: i64) -> Node {
    node("C").text("OO").int64(child).int64(parent)
}

pub(super) fn op(child: i64, parent: i64, prop: &str) -> Node {
    node("C").text("OP").int64(child).int64(parent).text(prop)
}

pub(super) fn global_settings(unit_scale_factor: f64) -> Node {
    node("GlobalSettings").child(properties70(vec![p_scalar(
        "UnitScaleFactor",
        unit_scale_factor,
    )]))
}

pub(super) fn geometry(id: i64, name: &str, positions: Vec<f64>, pvi: Vec<i32>) -> Node {
    object("Geometry", id, name, "Mesh")
        .child(node("Vertices").arr_f64(positions))
        .child(node("PolygonVertexIndex").arr_i32(pvi))
}

pub(super) fn uv_layer(name: &str, uvs: Vec<f64>, index: Option<Vec<i32>>) -> Node {
    let reference = if index.is_some() {
        "IndexToDirect"
    } else {
        "Direct"
    };
    let mut layer = node("LayerElementUV")
        .int32(0)
        .child(node("Name").text(name))
        .child(node("MappingInformationType").text("ByPolygonVertex"))
        .child(node("ReferenceInformationType").text(reference))
        .child(node("UV").arr_f64(uvs));
    if let Some(index) = index {
        layer = layer.child(node("UVIndex").arr_i32(index));
    }
    layer
}

pub(super) fn material_layer(mapping: &str, materials: Vec<i32>) -> Node {
    node("LayerElementMaterial")
        .int32(0)
        .child(node("MappingInformationType").text(mapping))
        .child(node("ReferenceInformationType").text("IndexToDirect"))
        .child(node("Materials").arr_i32(materials))
}

pub(super) fn model(id: i64, name: &str, sub_class: &str) -> Node {
    object("Model", id, name, sub_class)
}

pub(super) fn material(id: i64, name: &str) -> Node {
    object("Material", id, name, "")
}

pub(super) fn texture(id: i64, name: &str, relative: &str) -> Node {
    object("Texture", id, name, "").child(node("RelativeFilename").text(relative))
}

pub(super) fn skin_deformer(id: i64, name: &str) -> Node {
    object("Deformer", id, name, "Skin")
}

pub(super) fn cluster(id: i64, name: &str, indexes: Vec<i32>, weights: Vec<f64>) -> Node {
    object("Deformer", id, name, "Cluster")
        .child(node("Indexes").arr_i32(indexes))
        .child(node("Weights").arr_f64(weights))
}

pub(super) fn transform_link(m: Vec<f64>) -> Node {
    node("TransformLink").arr_f64(m)
}

pub(super) fn transform(m: Vec<f64>) -> Node {
    node("Transform").arr_f64(m)
}

// The 16 doubles FBX stores for a pure-translation matrix.
pub(super) fn flat_translation(t: [f64; 3]) -> Vec<f64> {
    let mut m = vec![0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m
}

pub(super) fn anim_stack(id: i64, name: &str) -> Node {
    object("AnimationStack", id, name, "")
}

pub(super) fn anim_layer(id: i64, name: &str) -> Node {
    object("AnimationLayer", id, name, "")
}

pub(super) fn anim_curve_node(id: i64, name: &str, defaults: [f64; 3]) -> Node {
    object("AnimationCurveNode", id, name, "").child(properties70(vec![
        p_scalar("d|X", defaults[0]),
        p_scalar("d|Y", defaults[1]),
        p_scalar("d|Z", defaults[2]),
    ]))
}

pub(super) fn anim_curve(id: i64, times: Vec<i64>, values: Vec<f32>) -> Node {
    object("AnimationCurve", id, "", "")
        .child(node("KeyTime").arr_i64(times))
        .child(node("KeyValueFloat").arr_f32(values))
}

// A whole document: object list, connection list and the unit scale that goes
// into GlobalSettings.
pub(super) struct Doc {
    pub(super) unit_scale_factor: f64,
    pub(super) objects: Vec<Node>,
    pub(super) connections: Vec<Node>,
}

impl Doc {
    // Append a child to the already-declared object with this id.
    pub(super) fn attach(&mut self, id: i64, child: Node) {
        let obj = self
            .objects
            .iter_mut()
            .find(|o| matches!(o.attrs.first(), Some(Attr::I64(v)) if *v == id))
            .expect("object id is declared");
        obj.children.push(child);
    }

    pub(super) fn replace_object(&mut self, id: i64, replacement: Node) {
        let obj = self
            .objects
            .iter_mut()
            .find(|o| matches!(o.attrs.first(), Some(Attr::I64(v)) if *v == id))
            .expect("object id is declared");
        *obj = replacement;
    }

    pub(super) fn drop_connection(&mut self, child: i64, parent: i64) {
        let before = self.connections.len();
        self.connections.retain(|c| {
            !matches!(
                (c.attrs.get(1), c.attrs.get(2)),
                (Some(Attr::I64(a)), Some(Attr::I64(b))) if *a == child && *b == parent
            )
        });
        assert!(
            self.connections.len() < before,
            "connection {child} -> {parent} is declared"
        );
    }

    pub(super) fn roots(self) -> Vec<Node> {
        vec![
            global_settings(self.unit_scale_factor),
            objects(self.objects),
            connections(self.connections),
        ]
    }

    pub(super) fn write(self) -> FbxFile {
        write(self.roots())
    }
}

pub(super) fn doc() -> Doc {
    Doc {
        unit_scale_factor: 100.0,
        objects: Vec::new(),
        connections: Vec::new(),
    }
}

pub(super) const GEOMETRY_ID: i64 = 100;
pub(super) const MESH_MODEL_ID: i64 = 200;
pub(super) const ROOT_BONE_ID: i64 = 300;
pub(super) const TIP_BONE_ID: i64 = 301;
pub(super) const SKIN_ID: i64 = 400;
pub(super) const ROOT_CLUSTER_ID: i64 = 401;
pub(super) const TIP_CLUSTER_ID: i64 = 402;

pub(super) const SECOND_GEOMETRY_ID: i64 = 110;
pub(super) const SECOND_MESH_MODEL_ID: i64 = 210;
pub(super) const SECOND_SKIN_ID: i64 = 410;
pub(super) const SECOND_CLUSTER_ID: i64 = 411;

// A skinned triangle bound to a two-joint chain: Root at the origin, Tip two
// units up. Control point 0 is fully Root-weighted, 2 fully Tip-weighted, and
// 1 is shared. The mesh node carries a +X geometric offset so the bind frame
// is not accidentally identity.
pub(super) fn two_bone_rig(unit_scale_factor: f64) -> Doc {
    Doc {
        unit_scale_factor,
        objects: vec![
            geometry(
                GEOMETRY_ID,
                "mesh",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3],
            ),
            model(MESH_MODEL_ID, "MeshNode", "Mesh").child(properties70(vec![p_vec3(
                "GeometricTranslation",
                [1.0, 0.0, 0.0],
            )])),
            model(ROOT_BONE_ID, "Root", "LimbNode"),
            model(TIP_BONE_ID, "Tip", "LimbNode"),
            skin_deformer(SKIN_ID, "Skin"),
            cluster(ROOT_CLUSTER_ID, "RootCluster", vec![0, 1], vec![1.0, 0.5])
                .child(transform_link(flat_translation([0.0, 0.0, 0.0])))
                .child(transform(flat_translation([0.0, 0.0, 0.0]))),
            cluster(TIP_CLUSTER_ID, "TipCluster", vec![1, 2], vec![0.5, 1.0])
                .child(transform_link(flat_translation([0.0, 2.0, 0.0])))
                .child(transform(flat_translation([0.0, -2.0, 0.0]))),
        ],
        connections: vec![
            oo(GEOMETRY_ID, MESH_MODEL_ID),
            oo(MESH_MODEL_ID, 0),
            oo(ROOT_BONE_ID, 0),
            oo(TIP_BONE_ID, ROOT_BONE_ID),
            oo(SKIN_ID, GEOMETRY_ID),
            oo(ROOT_CLUSTER_ID, SKIN_ID),
            oo(TIP_CLUSTER_ID, SKIN_ID),
            oo(ROOT_BONE_ID, ROOT_CLUSTER_ID),
            oo(TIP_BONE_ID, TIP_CLUSTER_ID),
        ],
    }
}

// [`two_bone_rig`] with a second skinned geometry ("hair") bound to the same
// Tip bone through its own skin deformer, so the file exposes two skinned
// meshes over one skeleton.
pub(super) fn two_part_rig(unit_scale_factor: f64) -> Doc {
    let mut doc = two_bone_rig(unit_scale_factor);
    doc.objects.extend([
        geometry(
            SECOND_GEOMETRY_ID,
            "hair",
            vec![0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            vec![0, 1, -3],
        ),
        model(SECOND_MESH_MODEL_ID, "HairNode", "Mesh"),
        skin_deformer(SECOND_SKIN_ID, "HairSkin"),
        cluster(
            SECOND_CLUSTER_ID,
            "HairCluster",
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
        )
        .child(transform_link(flat_translation([0.0, 2.0, 0.0])))
        .child(transform(flat_translation([0.0, -2.0, 0.0]))),
    ]);
    doc.connections.extend([
        oo(SECOND_GEOMETRY_ID, SECOND_MESH_MODEL_ID),
        oo(SECOND_MESH_MODEL_ID, 0),
        oo(SECOND_SKIN_ID, SECOND_GEOMETRY_ID),
        oo(SECOND_CLUSTER_ID, SECOND_SKIN_ID),
        oo(TIP_BONE_ID, SECOND_CLUSTER_ID),
    ]);
    doc
}

// [`two_part_rig`] written out with a one-second clip named "Wave", the shape
// the scene expansion reads: two skinned parts plus an animation stack.
pub(crate) fn two_part_rig_with_clip() -> FbxFile {
    const STACK: i64 = 500;
    const LAYER: i64 = 510;
    const CURVE_NODE: i64 = 520;
    const CURVE: i64 = 530;
    const ONE_SECOND: i64 = 46_186_158_000;

    let mut doc = two_part_rig(100.0);
    doc.objects.extend([
        anim_stack(STACK, "Wave").child(properties70(vec![
            p_time("LocalStart", 0),
            p_time("LocalStop", ONE_SECOND),
        ])),
        anim_layer(LAYER, "BaseLayer"),
        anim_curve_node(CURVE_NODE, "R", [0.0, 0.0, 0.0]),
        anim_curve(CURVE, vec![0, ONE_SECOND], vec![0.0, 45.0]),
    ]);
    doc.connections.extend([
        oo(LAYER, STACK),
        oo(CURVE_NODE, LAYER),
        op(CURVE, CURVE_NODE, "d|X"),
        op(CURVE_NODE, TIP_BONE_ID, "Lcl Rotation"),
    ]);
    doc.write()
}
