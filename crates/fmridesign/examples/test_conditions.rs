use fmridesign::events::{EventFactor, EventTerm};
use fmridesign::types::EventSpec;

fn main() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec!["A", "B", "A", "B", "A", "B"],
        vec![1.0; 6],
    )
    .unwrap();

    println!("Event name: {}", event.name());
    println!("Event levels: {:?}", event.levels());

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    println!("Term name: {}", term.name());

    // The conditions() method is part of TermInfo trait, which we can't easily access
    // from outside the crate. Let's check the design matrix column names instead.
}
