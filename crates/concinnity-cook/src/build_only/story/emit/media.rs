// Media files a story references, deduplicated by path in first-use order:
// each audio file becomes one AudioClip entry and each image (stage imagery
// and the optional menu backdrop) one Texture entry the sprites sample.
pub(super) struct MediaAssets {
    prefix: String,
    clips: Vec<(String, String)>,
    images: Vec<(String, String)>,
}

impl MediaAssets {
    pub(super) fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            clips: Vec::new(),
            images: Vec::new(),
        }
    }

    // The AudioClip asset name for an audio file path, allocating one on the
    // path's first use.
    pub(super) fn clip(&mut self, path: &str) -> String {
        allocate(&self.prefix, "clip", &mut self.clips, path)
    }

    // The Texture asset name for an image path, allocating one on the path's
    // first use.
    pub(super) fn image(&mut self, path: &str) -> String {
        allocate(&self.prefix, "img", &mut self.images, path)
    }

    // The AudioClip entries, then the Texture entries.
    pub(super) fn entries(&self) -> Vec<serde_json::Value> {
        let clips = self
            .clips
            .iter()
            .map(|(path, name)| (path, name, "AudioClip"));
        let images = self
            .images
            .iter()
            .map(|(path, name)| (path, name, "Texture"));
        clips
            .chain(images)
            .map(|(path, name, ty)| {
                serde_json::json!({
                    "name": name,
                    "type": ty,
                    "args": { "source": path }
                })
            })
            .collect()
    }
}

fn allocate(prefix: &str, kind: &str, named: &mut Vec<(String, String)>, path: &str) -> String {
    if let Some((_, name)) = named.iter().find(|(p, _)| p == path) {
        return name.clone();
    }
    let name = format!("{}_{}{}", prefix, kind, named.len());
    named.push((path.to_string(), name.clone()));
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_path_reuses_its_name() {
        let mut media = MediaAssets::new("s");
        assert_eq!(media.clip("a.ogg"), "s_clip0");
        assert_eq!(media.image("a.png"), "s_img0");
        assert_eq!(media.clip("b.ogg"), "s_clip1");
        assert_eq!(media.clip("a.ogg"), "s_clip0");
        let entries = media.entries();
        let types: Vec<_> = entries
            .iter()
            .map(|e| e["type"].as_str().unwrap())
            .collect();
        assert_eq!(types, ["AudioClip", "AudioClip", "Texture"]);
        assert_eq!(entries[2]["args"]["source"], "a.png");
    }
}
