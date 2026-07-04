use super::*;

impl StorySystem {
    // Arrive on the current page: run its variable ops, fill the stage, fire
    // its one-shots, log it in the backlog, and auto-save.
    pub(super) fn apply_page(&mut self, ctx: &mut PipelineContext) {
        let page = self.story.nodes[self.node].pages[self.page].clone();
        self.apply_ops(&page.ops);
        self.render_page(ctx);
        let entry = match &page.speaker {
            Some(s) => format!("{}: {}", s.name, page.text),
            None => page.text.clone(),
        };
        self.history.push(entry);
        if self.history.len() > BACKLOG_ENTRIES {
            self.history.remove(0);
        }
        for sound in &page.sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }
        self.persist_position(ctx);
    }

    // Fill the stage for the current page: name plate, dialogue reveal,
    // backdrop and portraits, page music. No arrival side effects (flag ops,
    // one-shots, the auto-save), so a hot-reload can re-render in place;
    // re-playing the page's music is a same-key no-op.
    pub(super) fn render_page(&mut self, ctx: &mut PipelineContext) {
        let (name_id, text_id) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (ids.name, ids.text)
        };
        let page = self.story.nodes[self.node].pages[self.page].clone();

        let (speaker, color) = match &page.speaker {
            Some(s) => (s.name.clone(), s.color),
            None => (String::new(), [1.0, 1.0, 1.0]),
        };
        set_label(ctx, name_id, |l| {
            l.content = speaker;
            l.color = color;
        });

        self.typewriter = Typewriter {
            full: page.text.chars().collect(),
            shown: 0,
            budget: 0.0,
        };
        if self.story.text_speed <= 0.0 {
            self.typewriter.shown = self.typewriter.full.len();
        }
        let text = self.typewriter.text();
        set_label(ctx, text_id, |l| l.content = text);

        apply_stage(
            ctx,
            self.ids.as_ref().expect("resolved at init"),
            &page.stage,
        );
        play(ctx, page.music, CueKind::Music);

        // A skip run reveals instantly; otherwise the reveal restarts the
        // auto/skip pacing clock.
        if self.skip {
            self.typewriter.shown = self.typewriter.full.len();
            let text = self.typewriter.text();
            let text_id = self.ids.as_ref().expect("resolved at init").text;
            set_label(ctx, text_id, |l| l.content = text);
        }
        self.mode_timer = 0.0;
        self.render_quick_row(ctx);
    }

    // Fill (or clear) the quick-row control labels; engaged modes read gold.
    pub(super) fn render_quick_row(&mut self, ctx: &mut PipelineContext) {
        let Some(ids) = self.ids.as_ref() else { return };
        let rows = [
            (ids.log_label, "Log", false),
            (ids.auto_label, "Auto", self.auto),
            (ids.skip_label, "Skip", self.skip),
            (ids.save_label, "Save", false),
        ];
        for (id, text, active) in rows {
            set_label(ctx, id, |l| {
                l.content = text.to_string();
                l.color = if active { QUICK_ACTIVE } else { QUICK_IDLE };
            });
        }
    }

    pub(super) fn clear_quick_row(&mut self, ctx: &mut PipelineContext) {
        let Some(ids) = self.ids.as_ref() else { return };
        for id in [
            ids.log_label,
            ids.auto_label,
            ids.skip_label,
            ids.save_label,
        ] {
            set_label(ctx, id, |l| l.content.clear());
        }
        set_sprite(ctx, ids.marker, |s| s.tint[3] = 0.0);
    }

    // Arrive on the current node's choice menu: run its variable ops, show
    // the menu, and fire its one-shots. A skip run stops at a menu.
    pub(super) fn enter_choice(&mut self, ctx: &mut PipelineContext) {
        let node = self.story.nodes[self.node].clone();
        self.apply_ops(&node.choice_ops);
        self.skip = false;
        self.render_choice(ctx);
        for sound in &node.choice_sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }
    }

    // Show the current node's choice menu over its stage dressing; stage
    // clicks are inert until an option is picked. Gated options are left off
    // the menu, and the button slots fill from the visible options in order.
    // No arrival side effects, so a hot-reload can re-render an open menu.
    pub(super) fn render_choice(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = true;
        let node = self.story.nodes[self.node].clone();
        self.menu = self.visible_choices(self.node);
        let menu = self.menu.clone();
        let ids = self.ids.as_ref().expect("resolved at init");

        apply_stage(ctx, ids, &node.choice_stage);
        play(ctx, node.choice_music, CueKind::Music);

        set_label(ctx, ids.name, |l| l.content.clear());
        set_label(ctx, ids.text, |l| l.content.clear());
        // Hidden stage furniture renders nothing (zero alpha, empty text)
        // rather than relying on `visible`: view re-activation (a pause
        // overlay dismissing back to the stage) force-shows every member.
        set_sprite(ctx, ids.dialog_box, |s| s.tint = [0.0, 0.0, 0.0, 0.0]);
        self.clear_quick_row(ctx);
        let ids = self.ids.as_ref().expect("resolved at init");
        let boxes = ids.option_boxes.clone();
        for (i, label_id) in ids.options.iter().enumerate() {
            let occupied = menu.get(i).is_some();
            set_sprite(ctx, boxes.get(i).copied(), |s| {
                s.visible = true;
                s.tint = if occupied {
                    CHOICE_BOX_TINT
                } else {
                    [
                        CHOICE_BOX_TINT[0],
                        CHOICE_BOX_TINT[1],
                        CHOICE_BOX_TINT[2],
                        0.0,
                    ]
                };
            });
            match menu.get(i).map(|&c| &node.choices[c]) {
                Some(choice) => {
                    let text = choice.label.clone();
                    // The build authors the label centered around MENU_CENTER_X
                    // (real-metric alignment), so the story only fills the text.
                    set_label(ctx, Some(*label_id), |l| {
                        l.content = text;
                        l.visible = true;
                        l.x = MENU_CENTER_X;
                    });
                }
                None => set_label(ctx, Some(*label_id), |l| l.content.clear()),
            }
        }
    }

    // Put the stage back into page mode (idempotent).
    pub(super) fn exit_choice_ui(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = false;
        let ids = self.ids.as_ref().expect("resolved at init");
        set_sprite(ctx, ids.dialog_box, |s| s.tint = [0.0, 0.0, 0.0, 0.55]);
        let boxes = ids.option_boxes.clone();
        for box_id in boxes {
            set_sprite(ctx, Some(box_id), |s| s.tint[3] = 0.0);
        }
        for label_id in &ids.options {
            set_label(ctx, Some(*label_id), |l| l.content.clear());
        }
    }
}
