use clap::Parser;
use clap_stdin::FileOrStdin;
use weiqi_pattern::{patterns_from_variation, variations_from_sgf};

#[derive(Parser, Debug)]
struct Args {
    #[clap(default_value = "-")]
    sgf_input: FileOrStdin,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let sgf_text = args.sgf_input.contents()?;
    let variations = variations_from_sgf(&sgf_text).unwrap();
    for variation in variations {
        let patterns = match patterns_from_variation(&variation) {
            Ok(patterns) => patterns,
            Err((patterns, e)) => {
                eprintln!(
                    "error {:?} occured while parsing variation: {:?}",
                    e, variation
                );
                patterns
            }
        };
        for pattern in patterns {
            println!("{}", pattern.repr());
        }
    }
    Ok(())
}
