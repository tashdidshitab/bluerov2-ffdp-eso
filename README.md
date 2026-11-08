# FFDP-ESO: lightweight feedforward + extended-state-observer control of the BlueROV2 Heavy

Simulation study (dive-in and lawnmower motions) of **FFDP-ESO** and baselines (PID, PID+FF, SMC, ADRC, MPC+ESO), with a delay predictor, online delay estimation, a delayed-measurement Kalman filter and fault-adaptive thrust reallocation.
**Everything here is simulation. No hardware or pool data.** Vehicle parameters are literature-style placeholders, not identified from a real vehicle.

Supplementary material (code, data and result files) for the associated paper. All reported numbers come from the files in `results/`.

## Layout
| Path | Content |
|---|---|
| `bluerov2_heavy_dataset.py` | 6-DOF vectorised simulator used to generate the synthetic dataset; defines `NOMINAL` parameters and the 6x8 allocation matrix |
| `controller_study.py` | Main study: references, all controllers and wrappers, simulator `sim(...)`, tuning (`tune`) and evaluation (`eval`, `stress`, `delaytest`, ...) |
| `ext_common.py` | Shared helpers (gains loading, bootstrap CIs, survey metrics) |
| `*_eval.py`, `mission_eval.py`, `nav_fail_*.py`, `extra*.py` | Extended studies (survey, ocean, dive, navigation failures, mission) |
| `mpc_tune.py`, `mpc_eval.py` | MPC+ESO baseline |
| `indep_sim.py`, `tune_indep.py`, `valid_*.py` | Independently coded plant (L1, L2) and validation |
| `twin_*.py` | Digital-twin-style case studies |
| `ft_final.py`, `guide_final.py`, `delay_est_eval*.py`, `hifi_eval.py`, `res_eval.py`, `nav_only.py` | Fault tolerance, recovery guidance (negative), online delay estimation, unmodeled-effects plant, learned residual (negative) |
| `make_report*.py`, `fig_vehicle.py` | Figures (`paper/`) and LaTeX tables |
| `bench/` | C benchmark of per-step controller cost |
| `results/` | CSV/JSON/log outputs and tuned gains (`tuned_*.json`) |
| `paper/` | Figures (PNG+PDF) and LaTeX tables |

Controller names in code: `FFDPE` = **FFDP-ESO** (proposed), `FFDP` = FFDP with integral term (FFDP-I), `FDP` = ablation without feedforward, `PIDFF` = PID+FF, `MPCE` = MPC+ESO. Suffixes: `_DC`/`_DCA` delay predictor (nav / nav+attitude), `_DCE`/`_DCN` estimated delay, `_FT`/`_FTO` fault-tolerant (detected / oracle), `_KF` Kalman filter, `_NF` fallback.

## Setup
Python 3.10+ with `pip install -r requirements.txt`. Run from the repository root (scripts use relative `results/` and `paper/` paths).

## Reproduce
```bash
python controller_study.py tune                      # ~12 min, results/tuned_gains.json
python controller_study.py eval                      # ~4 min, nominal + fault evaluation
python controller_study.py stress                    # ~25 min
python controller_study.py tune --delay 6 --controllers PID PIDFF FFDP ADRC FFDPE PIDFF_DCA FFDP_DCA ADRC_DCA FFDPE_DCA --out results/tuned_gains_delay0p3.json
python controller_study.py delaytest2 --n 200
python ft_final.py; python delay_est_eval.py PIDFF FFDPE; python hifi_eval.py
python survey_eval.py; python ocean_eval.py; python dive_eval.py; python nav_fail_eval.py; python mission_eval.py
python mpc_tune.py; python mpc_eval.py
python tune_indep.py; python valid_eval.py; python valid_delay.py; python valid_fault.py
python twin_analysis.py; python twin_monitor.py; python twin_figures.py
python make_report.py; python make_report2.py; python make_report3.py      # figures + tables
cd bench && gcc -O2 -o bench_ctrl bench_ctrl.c -lm && ./bench_ctrl
```
Random seeds are fixed. Runtimes are from the sandbox used for the study and will differ. Some long scripts may need to be run in the background.

## Not included (size)
The 10,000-episode synthetic dataset (about 594 MB), per-episode `.npz` logs and raw intermediate results are not in this repository. Regenerate the dataset with `python bluerov2_heavy_dataset.py --episodes 10000 --duration 30 --out dataset` (see `README_dataset.md`), or host the original archives on Zenodo/Releases.

## Honest limits
- Simulation only; the independent plant was written by the same author, and its L2 parameters are recalled from the literature and unverified.
- The delay predictor does not transfer to the independent plants; the advantage of FFDP-ESO over ADRC does not hold outside the study simulator; MPC+ESO is not better than FFDP-ESO.
- `results/` contains a few earlier intermediate files (for example `delay_tests.csv`, superseded by `delay_tests2.csv`, and `guidance_results.csv`, a negative result).

## License
No license has been chosen yet; add one (for example MIT) before making the repository public.
