//! Test SamplingFrame API
use fmri_design::sampling::{multi_block_sampling_frame, SamplingFrame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sf = multi_block_sampling_frame(&[50, 75, 100], &[2.0], None)?;

    // Try different method names to see what works
    println!("Number of scans: {}", sf.n_scans());
    println!("Number of blocks: {}", sf.n_blocks());

    // Try to get block lengths - let's see what methods exist
    // println!("Block lengths: {:?}", sf.block_lengths());
    // println!("Blocklens: {:?}", sf.blocklens());
    // println!("Block lens: {:?}", sf.block_lens());

    Ok(())
}
