//! Vignette 03: Event Models
//! ==========================
//!
//! Port of R's `a_04_event_models.Rmd`.
//!
//! This vignette shows how to construct event models of varying complexity:
//! single-factor designs, multi-run designs, parametric modulation with
//! continuous variables, interaction terms, and different event types.
//!
//! Run with:
//!   cargo run --example vignette_03_event_models

use fmridesign::core::EventModel;
use fmridesign::events::{EventFactor, EventTerm, EventVariable};
use fmridesign::sampling::{multi_block_sampling_frame, uniform_sampling_frame};
use fmridesign::types::EventSpec;
use fmridesign::visualization::{correlation_summary, design_summary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vignette 03: Event Models ===\n");

    // ---------------------------------------------------------------
    // 1. Single-factor categorical design
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition), data, block = ~run, sframe)
    //
    // The simplest design: one categorical factor with multiple levels.
    println!("--- Example 1: Single-factor categorical design ---\n");

    let sf = uniform_sampling_frame(200, 2.0, None)?;

    // Create events: 3 conditions, 8 trials each, spaced 10s apart
    let mut onsets = Vec::new();
    let mut conditions = Vec::new();
    let mut durations = Vec::new();

    let levels = ["face", "house", "scrambled"];
    for (i, &cond) in levels.iter().enumerate() {
        for trial in 0..8 {
            onsets.push(5.0 + (i * 8 + trial) as f64 * 10.0);
            conditions.push(cond);
            durations.push(0.5); // brief 500ms stimulus
        }
    }

    let factor = EventFactor::new(
        "stimulus",
        onsets.clone(),
        conditions.clone(),
        durations.clone(),
    )?;
    println!(
        "  EventFactor: '{}' with {} levels, {} events",
        factor.name,
        factor.n_levels(),
        factor.onsets.len()
    );
    println!("  Levels: {:?}", factor.levels);

    let term = EventTerm::new(vec![Box::new(factor)], false)?;
    let model = EventModel::new(vec![Box::new(term)], sf.clone())?;

    let dm = model.generate_design_matrix()?;
    println!(
        "  Design matrix: {} x {} (rows x cols)",
        dm.nrows(),
        dm.ncols()
    );
    println!("  Columns: {:?}", model.column_names());
    println!();
    println!("{}", design_summary(&dm, &model.column_names()));
    println!("{}", correlation_summary(&dm, &model.column_names(), 0.20));

    // Show non-zero pattern
    println!("  Non-zero counts per column:");
    for (j, name) in model.column_names().iter().enumerate() {
        let nz = dm.column(j).iter().filter(|&&v| v.abs() > 1e-10).count();
        println!("    {:<15} {} non-zero values", name, nz);
    }
    println!();

    // ---------------------------------------------------------------
    // 2. Multi-run design
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition), data, block = ~run, sframe)
    //
    // When an experiment spans multiple runs, events are placed at their
    // absolute time positions within the concatenated timeline.
    println!("--- Example 2: Multi-run design ---\n");

    let sf_multi = multi_block_sampling_frame(&[100, 100], &[2.0], None)?;
    println!(
        "  Sampling frame: {} runs, {} total scans",
        sf_multi.n_blocks(),
        sf_multi.n_scans()
    );

    // Run 1: events in [0, 200) seconds
    // Run 2: events in [200, 400) seconds
    let mut mr_onsets = Vec::new();
    let mut mr_conditions = Vec::new();
    let mut mr_durations = Vec::new();

    for run in 0..2 {
        let offset = run as f64 * 200.0;
        for trial in 0..6 {
            mr_onsets.push(offset + 10.0 + trial as f64 * 30.0);
            mr_conditions.push(if trial % 2 == 0 { "go" } else { "nogo" });
            mr_durations.push(1.0);
        }
    }

    let mr_factor = EventFactor::new("response", mr_onsets, mr_conditions, mr_durations)?;
    let mr_term = EventTerm::new(vec![Box::new(mr_factor)], false)?;
    let mr_model = EventModel::new(vec![Box::new(mr_term)], sf_multi.clone())?;

    let mr_dm = mr_model.generate_design_matrix()?;
    println!("  Design matrix: {} x {}", mr_dm.nrows(), mr_dm.ncols());
    println!("  Columns: {:?}", mr_model.column_names());

    // Show that events appear in both halves
    let half = mr_dm.nrows() / 2;
    let go_col = 0;
    let run1_nz = mr_dm
        .column(go_col)
        .iter()
        .take(half)
        .filter(|&&v| v > 0.0)
        .count();
    let run2_nz = mr_dm
        .column(go_col)
        .iter()
        .skip(half)
        .filter(|&&v| v > 0.0)
        .count();
    println!(
        "  '{}' events: {} in run 1, {} in run 2",
        mr_model.column_names()[go_col],
        run1_nz,
        run2_nz
    );
    println!();

    // ---------------------------------------------------------------
    // 3. Parametric modulation with a continuous variable
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition) + hrf(condition):scale(rt), ...)
    //
    // Parametric modulators allow trial-by-trial variation in the amplitude
    // of the hemodynamic response. Here we model reaction time (RT) as a
    // continuous modulator.
    println!("--- Example 3: Parametric modulation ---\n");

    let pm_sf = uniform_sampling_frame(150, 2.0, None)?;

    // Create onsets and RT values
    let pm_onsets: Vec<f64> = (0..15).map(|i| 5.0 + i as f64 * 18.0).collect();
    let pm_durations = vec![1.0; 15];
    let rt_values: Vec<f64> = vec![
        0.45, 0.52, 0.38, 0.61, 0.44, 0.55, 0.42, 0.58, 0.39, 0.50, 0.47, 0.63, 0.41, 0.56, 0.43,
    ];

    println!("  {} trials with RT values", pm_onsets.len());
    println!(
        "  RT range: [{:.2}, {:.2}]",
        rt_values.iter().cloned().fold(f64::INFINITY, f64::min),
        rt_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // EventVariable captures continuous values
    // By default, values are centered (mean-subtracted)
    let rt_event = EventVariable::new(
        "RT",
        pm_onsets.clone(),
        rt_values.clone(),
        pm_durations.clone(),
    )?;

    println!(
        "  EventVariable '{}': centered={}, scaled={}",
        rt_event.name, rt_event.center, rt_event.scale
    );
    println!(
        "  Raw mean: {:.4}, transformed mean: {:.4}",
        rt_event.mean(),
        rt_event.values.mean().unwrap_or(0.0)
    );

    // Create a combined model with both the categorical and parametric terms
    // Term 1: categorical onset indicator
    let pm_conditions: Vec<&str> = vec!["task"; 15];
    let pm_factor = EventFactor::new(
        "task",
        pm_onsets.clone(),
        pm_conditions,
        pm_durations.clone(),
    )?;
    let main_term = EventTerm::new(vec![Box::new(pm_factor)], false)?;

    // Term 2: continuous parametric modulator
    let rt_term = EventTerm::new(vec![Box::new(rt_event)], false)?;

    let pm_model = EventModel::new(vec![Box::new(main_term), Box::new(rt_term)], pm_sf.clone())?;

    let pm_dm = pm_model.generate_design_matrix()?;
    println!("\n  Design matrix: {} x {}", pm_dm.nrows(), pm_dm.ncols());
    println!("  Columns: {:?}", pm_model.column_names());

    // The parametric column contains centered RT values at event times
    let rt_col_idx = 1; // RT column
    let rt_col = pm_dm.column(rt_col_idx);
    let rt_nonzero: Vec<f64> = rt_col
        .iter()
        .filter(|&&v| v.abs() > 1e-10)
        .cloned()
        .collect();
    println!(
        "  RT column: {} non-zero values, range [{:.4}, {:.4}]",
        rt_nonzero.len(),
        rt_nonzero.iter().cloned().fold(f64::INFINITY, f64::min),
        rt_nonzero.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // ---------------------------------------------------------------
    // 4. Centering and scaling options for continuous variables
    // ---------------------------------------------------------------
    println!("--- Example 4: Centering and scaling options ---\n");

    let raw_values = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let ev_onsets = vec![5.0, 15.0, 25.0, 35.0, 45.0];
    let ev_durs = vec![1.0; 5];

    // No centering, no scaling
    let ev_raw = EventVariable::new(
        "raw",
        ev_onsets.clone(),
        raw_values.clone(),
        ev_durs.clone(),
    )?
    .with_center(false)
    .with_scale(false);
    println!(
        "  center=false, scale=false: values = {:?}",
        ev_raw.values.to_vec()
    );

    // Centered (default)
    let ev_centered = EventVariable::new(
        "centered",
        ev_onsets.clone(),
        raw_values.clone(),
        ev_durs.clone(),
    )?;
    println!(
        "  center=true,  scale=false: values = {:?}",
        ev_centered.values.to_vec()
    );

    // Centered and scaled (z-score)
    let ev_scaled = EventVariable::new(
        "scaled",
        ev_onsets.clone(),
        raw_values.clone(),
        ev_durs.clone(),
    )?
    .with_scale(true);
    println!(
        "  center=true,  scale=true:  values = {:?}",
        ev_scaled
            .values
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );
    println!();

    // ---------------------------------------------------------------
    // 5. Interaction terms (categorical x categorical)
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(task:stimulus), ...)
    //
    // When two factors are crossed, their interaction creates a column
    // for every combination of levels.
    println!("--- Example 5: Interaction terms ---\n");

    let mut ix_onsets = Vec::new();
    let mut ix_task = Vec::new();
    let mut ix_stim = Vec::new();
    let mut ix_durs = Vec::new();

    let tasks = ["easy", "hard"];
    let stims = ["word", "nonword"];
    let mut t = 5.0;

    for _ in 0..4 {
        for &task in &tasks {
            for &stim in &stims {
                ix_onsets.push(t);
                ix_task.push(task);
                ix_stim.push(stim);
                ix_durs.push(1.0);
                t += 12.0;
            }
        }
    }

    let task_factor = EventFactor::new("task", ix_onsets.clone(), ix_task, ix_durs.clone())?;
    let stim_factor = EventFactor::new("stimulus", ix_onsets.clone(), ix_stim, ix_durs.clone())?;

    println!("  Task levels:     {:?}", task_factor.levels);
    println!("  Stimulus levels: {:?}", stim_factor.levels);
    println!("  Total events:    {}", ix_onsets.len());

    // Create interaction term: task x stimulus
    let ix_term = EventTerm::new(
        vec![Box::new(task_factor.clone()), Box::new(stim_factor.clone())],
        true, // interaction = true
    )?
    .with_interaction(true);

    println!("  Interaction name: '{}'", ix_term.name());
    println!("  Is categorical:   {}", ix_term.is_categorical());

    let ix_sf = uniform_sampling_frame(200, 2.0, None)?;
    let ix_model = EventModel::new(vec![Box::new(ix_term)], ix_sf)?;

    let ix_dm = ix_model.generate_design_matrix()?;
    println!("  Design matrix:    {} x {}", ix_dm.nrows(), ix_dm.ncols());
    println!("  Columns:          {:?}", ix_model.column_names());
    println!();

    // ---------------------------------------------------------------
    // 6. Mixed interaction (categorical x continuous)
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition):scale(rt), ...)
    //
    // Mixing a factor with a continuous variable creates one parametric
    // modulator column per level of the factor.
    println!("--- Example 6: Mixed interaction (categorical x continuous) ---\n");

    let mix_onsets: Vec<f64> = (0..12).map(|i| 5.0 + i as f64 * 15.0).collect();
    let mix_conds: Vec<&str> = (0..12)
        .map(|i| if i % 2 == 0 { "A" } else { "B" })
        .collect();
    let mix_rts: Vec<f64> = vec![
        0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.45, 0.65, 0.35, 0.75, 0.42, 0.58,
    ];
    let mix_durs = vec![1.0; 12];

    let mix_factor = EventFactor::new("cond", mix_onsets.clone(), mix_conds, mix_durs.clone())?;
    let mix_variable =
        EventVariable::new("rt", mix_onsets.clone(), mix_rts, mix_durs.clone())?.with_center(false);

    let mix_term = fmridesign::events::event_term::create_interaction(
        vec![Box::new(mix_factor), Box::new(mix_variable)],
        Some("cond:rt".to_string()),
    );

    println!("  Term name:      '{}'", mix_term.name());
    println!("  Is mixed:       {}", mix_term.is_mixed());

    let mix_sf = uniform_sampling_frame(150, 2.0, None)?;
    let mix_model = EventModel::new(vec![Box::new(mix_term)], mix_sf)?;

    let mix_dm = mix_model.generate_design_matrix()?;
    println!("  Design matrix:  {} x {}", mix_dm.nrows(), mix_dm.ncols());
    println!("  Columns:        {:?}", mix_model.column_names());

    // Show that condition A and B columns have different patterns
    for (j, name) in mix_model.column_names().iter().enumerate() {
        let col = mix_dm.column(j);
        let nz = col.iter().filter(|&&v| v.abs() > 1e-10).count();
        let sum: f64 = col.iter().filter(|&&v| v.abs() > 1e-10).sum();
        println!("    {:<20} non-zero={}, sum={:.4}", name, nz, sum);
    }
    println!();

    // ---------------------------------------------------------------
    // 7. Multiple terms in one model
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition) + hrf(rt), ...)
    //
    // A design can have many terms. Each contributes columns to the
    // final design matrix.
    println!("--- Example 7: Multiple terms in one model ---\n");

    let mt_sf = uniform_sampling_frame(180, 2.0, None)?;

    // Term 1: categorical condition factor
    let mt_onsets: Vec<f64> = (0..12).map(|i| 8.0 + i as f64 * 20.0).collect();
    let mt_conds: Vec<&str> = (0..12)
        .map(|i| match i % 3 {
            0 => "X",
            1 => "Y",
            _ => "Z",
        })
        .collect();
    let mt_durs = vec![2.0; 12];
    let cond_factor = EventFactor::new("condition", mt_onsets.clone(), mt_conds, mt_durs.clone())?;
    let cond_term = EventTerm::new(vec![Box::new(cond_factor)], false)?;

    // Term 2: continuous parametric regressor (difficulty rating)
    let difficulty: Vec<f64> = vec![3.0, 7.0, 5.0, 8.0, 2.0, 6.0, 4.0, 9.0, 1.0, 7.0, 3.0, 8.0];
    let diff_event =
        EventVariable::new("difficulty", mt_onsets.clone(), difficulty, mt_durs.clone())?;
    let diff_term = EventTerm::new(vec![Box::new(diff_event)], false)?;

    // Term 3: continuous parametric regressor (confidence)
    let confidence: Vec<f64> = vec![
        0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.95, 0.5, 0.85, 0.15,
    ];
    let conf_event = EventVariable::new("confidence", mt_onsets.clone(), confidence, mt_durs)?;
    let conf_term = EventTerm::new(vec![Box::new(conf_event)], false)?;

    let mt_model = EventModel::new(
        vec![
            Box::new(cond_term),
            Box::new(diff_term),
            Box::new(conf_term),
        ],
        mt_sf,
    )?;

    println!("  {}", mt_model);
    let mt_dm = mt_model.generate_design_matrix()?;
    let mt_cols = mt_model.column_names();
    println!("  Design matrix: {} x {}", mt_dm.nrows(), mt_dm.ncols());
    println!("  All columns:");
    for (i, name) in mt_cols.iter().enumerate() {
        let col = mt_dm.column(i);
        let nz = col.iter().filter(|&&v| v.abs() > 1e-10).count();
        println!("    [{:>2}] {:<20} ({} non-zero)", i, name, nz);
    }
    println!();

    // ---------------------------------------------------------------
    // 8. Using the EventModelBuilder
    // ---------------------------------------------------------------
    // The builder pattern provides a fluent API for constructing models.
    println!("--- Example 8: EventModelBuilder ---\n");

    let b_sf = uniform_sampling_frame(100, 2.0, None)?;

    let b_factor = EventFactor::new(
        "shape",
        vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
        vec!["circle", "square", "circle", "square", "circle", "square"],
        vec![0.5; 6],
    )?;
    let b_term = EventTerm::new(vec![Box::new(b_factor)], false)?;

    let builder_model = fmridesign::core::EventModelBuilder::new()
        .add_event(Box::new(b_term))
        .sampling_frame(b_sf)
        .name("Shape Perception")
        .build()?;

    println!("  Model: '{}'", builder_model.name);
    println!(
        "  Events: {}, Samples: {}, Columns: {}",
        builder_model.n_events(),
        builder_model.n_samples(),
        builder_model.n_columns()
    );
    println!("  Column names: {:?}", builder_model.column_names());
    println!();

    println!("=== Vignette 03 complete ===");
    Ok(())
}
