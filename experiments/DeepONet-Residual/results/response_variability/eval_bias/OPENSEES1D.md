# OpenSees 1-D vs Haskell ξ_soil vs OpenSees 2-D

Third arm after `HASKELL_XI.md`: a **literal 1-D OpenSees column** (Rayleigh, not hysteretic Haskell). Isolates damping *model shape* the same way that note isolated the ξ *value*.

Setup: `boundary_condition_type="1D"` simple-shear column, `uniform_soil_only` with ζ = H5 soil-mean `Damping_zeta`, Rayleigh matched at `(damping_freq_first, 10 Hz)` — the 2-D data-gen convention, not the 1-D validation `(f_0, 3f_0)`. Analysis `dt = 0.01` s (2-D solver step; stored H5 recorders are often 0.02 s), `hx = 1` m, `motion_t_shift = 0.5` s, 10 m rock buffer. **Anderson** is the same Gaussian-weighted $L_1$ on $\ln|\mathrm{TF}|$ as in `HASKELL_XI.md`.

Two within motions: **interface** is the soil–rock contact (Haskell `AF_within`); **pack base** is the $y=2$ m recorder used to build the OpenSees 2-D pack TFs. Outcrop is $|\mathrm{FAS}_{\mathrm{surf}}/(2 a_{\mathrm{incident}})|$. Mode $k$ is the peak in $[2(k-1)f_0,\,2k f_0]$ clipped to 0.1–10 Hz.

| Domain | n | Arm | Pearson | Anderson | $\Delta\ln A_1$ | $\Delta\ln A_2$ | $\Delta\ln A_3$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| iid | 150 | Haskell ξ_soil vs 2-D (within) | 0.741 [0.616, 0.861] | 0.169 [0.114, 0.244] | 0.148 [-0.135, 0.450] | -0.307 [-0.670, 0.064] | -0.394 [-0.657, -0.081] |
| iid | 150 | OpenSees 1-D vs 2-D (pack base) | 0.776 [0.648, 0.898] | 0.151 [0.094, 0.231] | 0.121 [-0.143, 0.427] | 0.245 [-0.083, 0.560] | 0.113 [-0.003, 0.352] |
| iid | 150 | Haskell ξ_soil vs OpenSees 1-D (within) | 0.981 [0.972, 0.987] | 0.017 [0.012, 0.025] | -0.017 [-0.023, -0.011] | 0.559 [0.328, 0.677] | 0.527 [0.264, 0.693] |
| iid | 150 | Haskell ξ_soil vs OpenSees 1-D (outcrop) | 0.754 [0.624, 0.878] | 0.186 [0.142, 0.284] | -0.764 [-0.995, -0.549] | -1.710 [-2.444, -1.286] | -0.827 [-1.542, -0.502] |
| dipping | 144 | Haskell ξ_soil vs 2-D (within) | 0.674 [0.570, 0.789] | 0.178 [0.118, 0.873] | 0.476 [0.225, 0.628] | -0.087 [-0.359, 0.119] | -0.720 [-1.037, -0.503] |
| dipping | 144 | OpenSees 1-D vs 2-D (pack base) | 0.835 [0.762, 0.896] | 0.142 [0.091, 0.284] | 0.467 [0.221, 0.623] | 0.545 [0.254, 0.769] | 0.106 [-0.076, 0.279] |
| dipping | 144 | Haskell ξ_soil vs OpenSees 1-D (within) | 0.985 [0.981, 0.988] | 0.011 [0.010, 0.016] | -0.015 [-0.017, -0.012] | 0.464 [0.353, 0.567] | 0.344 [0.131, 0.486] |
| dipping | 144 | Haskell ξ_soil vs OpenSees 1-D (outcrop) | 0.712 [0.486, 0.842] | 0.200 [0.162, 0.305] | -0.697 [-0.925, -0.571] | -1.940 [-2.509, -1.214] | -1.152 [-1.633, -0.703] |

## Reading

- **Haskell vs OpenSees 1-D, within, mode 1** is *not* the $\Delta\ln A \approx -0.22$ from the 1-D validation column. That column matched Rayleigh at $(f_0, 3f_0)$. Here $f_2=10$ Hz (the 2-D data-gen convention), so mode 1 sits near a match frequency and the two 1-D models agree (Pearson $\sim 0.98$, $\Delta\ln A_1 \sim -0.02$). Rayleigh sags between the two match frequencies, so OpenSees 1-D is *taller* than hysteretic Haskell at modes 2–3.
- **Outcrop** is a different story: OpenSees 1-D peaks fall well below Haskell (mode-1 $\Delta\ln A$ of order $-0.7$), matching the attached 1-D validation outcrop panel. Pack 2-D TFs are *within* motion, so outcrop numerics do not enter the 0.77 / 0.69 Pearson numbers.
- **OpenSees 1-D vs OpenSees 2-D** (pack-base within) is the leftover after matching the FE / Rayleigh scheme. That is the fraction that can still be attributed to genuine spatial content before GINO. 2-D itself uses `global_avg` on a GRF, so this 1-D arm (uniform soil ζ) is not a bit-identical twin. Residual GINO is not retrained.

- **iid:** Haskell vs 1-D OpenSees Pearson (within) 0.981 [0.972, 0.987], mode-1 $\Delta\ln A$ -0.017 [-0.023, -0.011] (outcrop Pearson 0.754 [0.624, 0.878], $\Delta\ln A_1$ -0.764 [-0.995, -0.549]). OpenSees 1-D vs 2-D Pearson 0.776 [0.648, 0.898] vs Haskell vs 2-D 0.741 [0.616, 0.861].
- **dipping:** Haskell vs 1-D OpenSees Pearson (within) 0.985 [0.981, 0.988], mode-1 $\Delta\ln A$ -0.015 [-0.017, -0.012] (outcrop Pearson 0.712 [0.486, 0.842], $\Delta\ln A_1$ -0.697 [-0.925, -0.571]). OpenSees 1-D vs 2-D Pearson 0.835 [0.762, 0.896] vs Haskell vs 2-D 0.674 [0.570, 0.789]. Dipping 2-D high-$f$ rise is shared with OpenSees 1-D and absent from Haskell — that is most of the 0.67→0.84 Pearson lift.
