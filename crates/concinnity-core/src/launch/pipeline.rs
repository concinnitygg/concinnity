/// Whether a run overlaps simulation and rendering on separate threads (the
/// default) or steps both serially on one thread (the editor's mode, and
/// `cn run --serial` for A/B comparison and as an escape hatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    /// Simulation and rendering overlap on separate threads.
    #[default]
    Pipelined,
    /// Simulation and rendering step serially on the main thread.
    Serial,
}

impl PipelineMode {
    /// The mode a run takes: a world that `renders` nothing has no render half
    /// to overlap with, so it steps serially.
    pub fn effective(self, renders: bool) -> Self {
        if renders { self } else { Self::Serial }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_rendering_world_pipelines() {
        assert_eq!(
            PipelineMode::Pipelined.effective(true),
            PipelineMode::Pipelined
        );
        assert_eq!(
            PipelineMode::Pipelined.effective(false),
            PipelineMode::Serial
        );
        assert_eq!(PipelineMode::Serial.effective(true), PipelineMode::Serial);
    }
}
