use mtg_kernel::rl_session::KernelRlJsonlServerV1;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut server = KernelRlJsonlServerV1::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                eprintln!("stdin read failed: {err}");
                std::process::exit(1);
            }
        };
        let response = server.handle_line(&line);
        if let Err(err) = writeln!(stdout, "{response}") {
            eprintln!("stdout write failed: {err}");
            std::process::exit(1);
        }
        if let Err(err) = stdout.flush() {
            eprintln!("stdout flush failed: {err}");
            std::process::exit(1);
        }
    }
}
