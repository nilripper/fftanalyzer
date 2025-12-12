use eframe::egui;

pub const PLATINUM_BG: egui::Color32 = egui::Color32::from_rgb(212, 208, 200);
pub const PLATINUM_DARK: egui::Color32 = egui::Color32::from_rgb(128, 128, 128);

pub fn setup_global_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    //
    // Set global background fill colors.
    //
    style.visuals.panel_fill = PLATINUM_BG;
    style.visuals.window_fill = PLATINUM_BG;

    //
    // Remove widget rounding to match UI aesthetic.
    //
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::ZERO;
    style.visuals.widgets.active.rounding = egui::Rounding::ZERO;
    style.visuals.widgets.inactive.rounding = egui::Rounding::ZERO;
    style.visuals.widgets.hovered.rounding = egui::Rounding::ZERO;

    ctx.set_style(style);
}

/// Draws a simplified menu bar.
pub fn draw_menu_bar(ui: &mut egui::Ui, algorithm_name: &str) {
    egui::TopBottomPanel::top("menubar").show_inside(ui, |ui| {
        ui.visuals_mut().widgets.noninteractive.bg_fill = PLATINUM_BG;
        ui.horizontal(|ui| {
            //
            // Application title.
            //
            ui.label(egui::RichText::new("fftanalyzer").strong());

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                //
                // Algorithm information label.
                //
                ui.label(egui::RichText::new(algorithm_name).italics().size(10.0));
            });
        });
    });
}

/// Draws a window styled with the "Platinum" retro frame.
pub fn draw_platinum_window<F: FnOnce(&mut egui::Ui)>(ui: &mut egui::Ui, title: &str, content: F) {
    let frame = egui::Frame::none()
        .fill(PLATINUM_BG)
        .stroke(egui::Stroke::new(1.0, egui::Color32::BLACK))
        .inner_margin(2.0);

    frame.show(ui, |ui| {
        //
        // Title bar region.
        //
        let title_height = 18.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), title_height),
            egui::Sense::hover(),
        );

        //
        // Title bar background with pinstripe effect.
        //
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgb(200, 200, 200));
        for i in (0..rect.width() as i32).step_by(2) {
            let x = rect.min.x + i as f32;
            ui.painter().line_segment(
                [
                    egui::Pos2::new(x, rect.min.y),
                    egui::Pos2::new(x, rect.max.y),
                ],
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_premultiplied(255, 255, 255, 50),
                ),
            );
        }

        //
        // Centered title text.
        //
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            title,
            egui::FontId::proportional(14.0),
            egui::Color32::BLACK,
        );

        //
        // Content region.
        //
        ui.add_space(4.0);
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.0, PLATINUM_DARK)) // Inner bevel border.
            .inner_margin(6.0)
            .show(ui, content);
    });
}

/// Returns heatmap color (Black → Blue → Cyan → Green → Yellow → Red).
pub fn get_heatmap_color(val: f32) -> (u8, u8, u8) {
    if val < 0.2 {
        //
        // Black → Blue gradient.
        //
        return (0, 0, (val * 5.0 * 255.0) as u8);
    }
    if val < 0.4 {
        //
        // Blue → Cyan gradient.
        //
        return (0, ((val - 0.2) * 5.0 * 255.0) as u8, 255);
    }
    if val < 0.6 {
        //
        // Cyan → Green gradient.
        //
        return (0, 255, (255.0 - (val - 0.4) * 5.0 * 255.0) as u8);
    }
    if val < 0.8 {
        //
        // Green → Yellow gradient.
        //
        return (((val - 0.6) * 5.0 * 255.0) as u8, 255, 0);
    }

    //
    // Yellow → Red gradient.
    //
    (255, (255.0 - (val - 0.8) * 5.0 * 255.0) as u8, 0)
}
