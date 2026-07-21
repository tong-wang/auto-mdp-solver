# cases/ — auto-solve test cases

Each subfolder is one end-to-end run of the mdp-solver skill on a new
problem: IR + restatement (Phase A), generated domain + benchmarks + trained
policy (Phase B), and a README with the leaderboard. The purpose of this
directory is to grow the solver's capability envelope — every case should
stress something the pipeline hasn't handled before, and a case that forces
an `mdp_ir/` extension is a good case.

Trained artifacts (`results/`) are gitignored; each case README's commands
must reproduce them. A case whose domain proves broadly useful as a few-shot
exemplar can be promoted to `examples/` (add manifest row).
