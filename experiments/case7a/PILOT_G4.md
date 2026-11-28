# G4 is blocked on the pilot solves

Production N is not chosen. The 8-master OpenSees pilot has not been run, so the between-master variance of the paired width difference does not exist yet. This note is the procedure to apply once that variance is in hand. It does not pick a master count.

The local manifest is `experiments/GIFNO-VARDOMAIN/hpc/pilot_g4.csv`: 8 masters, widths 500/1000/1500/2000 m as crops of one 2000 m field, 3 seeds, gradient and layered fills, plus a CoV = 0 slice, `depth_m` = 250, explicit damping drawn independently of Vs. Record length is not in that CSV. `stampede3_pilot.slurm` times one 2000 m master at 200 s and at 400 s and does not start the 8-master run until `RECORD_S` is chosen from those timings and the job is resubmitted with `PILOT_CONFIRMED=1`.

## What the pilot must return before any N

For each harness band (0.1–0.4 Hz, 0.4–2.5 Hz, 2.5–10 Hz, and 0.1–10 Hz), each soil fill, and each wider crop W in {1000, 1500, 2000} m:

1. Score absolute g log-L1 on the solved records. g is log|TF| minus log|TF_1D| with the corrected outcrop.
2. Pair each wider crop with the 500 m crop of the same master, the same seed, and the same fill. The difference is log-L1(W) − log-L1(500 m).
3. Average those differences over the 3 seeds inside each master. The CoV = 0 slice is not part of this variance.
4. The independent sample is the master. `paired_bootstrap_group` is `master_id`. Width crops and stations are not resampled as if they were new geologies.

Let d_m be that seed-averaged paired difference for master m, and let s² be the sample variance of d_m across the 8 masters (divisor n − 1 = 7). Compute s² separately for every band, fill, and width contrast.

Also keep, for the other Phase 4 notes and not for this N: pad contamination on the scored stations, and the seed-to-seed spread of g inside a master.

## TOST once s² exists

Equivalence margin Δ = 0.02 in log-L1, per band. Two one-sided tests, each at α = 0.05. Power 0.8, so β = 0.2. The planning assumption is a true paired difference of 0. N is a number of masters.

Solve the smallest integer n ≥ 2 that satisfies

    n ≥ (t_{n−1, 1−α} + t_{n−1, 1−β})² × s² / Δ²

with t the Student-t quantile, Δ = 0.02, 1−α = 0.95, and 1−β = 0.80. Iterate n upward from 2. Do this for every band, fill, and width contrast, and take the maximum. That maximum is the candidate production master count. If it is unaffordable, shrink the claim. Either outcome is written only after the pilot variance exists.

A normal approximation, (z_{0.95} + z_{0.80})² ≈ 6.18, is a check on the iteration when n is large. The t iteration is the one that sets n, because s² will come from 8 masters.

The 3 seeds stay inside the master average. They do not multiply the sample size. Submitting a production campaign, and writing a lateral section from it, waits on this n.
