# Learning Across the Gap – Experiment Reproduction Code

Python reproduction of all experiments from:

> **"Learning Across the Gap: Hybrid Multi-armed Bandits with Heterogeneous Offline and Online Data"**  
> He, Wang, Liu, Wang, Kong — NeurIPS 2025

---

## File Structure

```
bandits/
├── environment.py           # Bradley-Terry model, data generation, env classes
├── hybucb_ar.py             # Algorithm 1: HybUCB-AR (dueling + offline absolute)
├── hybucb_elimra.py         # Algorithm 2: HybElimUCB-RA (stochastic + offline relative)
├── baselines.py             # RUCB, IF2, UCB, ETC, Thompson Sampling
├── experiments_synthetic.py # Synthetic experiment functions (Figures 1–5)
├── experiments_realworld.py # Real-world experiment functions (Figures 6–7)
├── plotting.py              # All plotting utilities
├── main.py                  # Main entry point (CLI)
└── results/                 # Output figures (auto-created)
```

---

## Quick Start

```bash
# Fast demo (5 trials, reduced T) – runs in ~1 minute
python main.py --mode demo

# Medium (20 trials) – good balance of speed and accuracy
python main.py --mode medium

# Full paper settings (100 trials, T=50000) – may take several hours
python main.py --mode full
```

### Run specific experiments

```bash
# Scalability figures (Figures 2 & 4)
python main.py --mode medium --exp hybucbar_scalability
python main.py --mode medium --exp hybelimucbra_scalability

# Parameter sensitivity (Figures 3 & 5)
python main.py --mode medium --exp sensitivity

# Main paper Figure 1
python main.py --mode medium --exp figure1

# Real-world experiments (Figures 6 & 7)
python main.py --mode medium --exp realworld
```

### With real datasets

```bash
# MovieLens-20M: download ratings.csv from https://grouplens.org/datasets/movielens/20m/
# Yelp Academic: download from https://www.yelp.com/dataset

python main.py --mode full --exp realworld \
  --movielens /path/to/ml-20m/ratings.csv \
  --yelp /path/to/yelp_academic_dataset_review.json
```

If datasets are not provided, the code automatically falls back to a
synthetic substitute with realistic rating statistics.

---

## Algorithm Details

### HybUCB-AR  (Algorithm 1, Section 4)
- **Setting**: Online dueling bandits + offline absolute (reward) data
- **Key idea**: Construct a MVUE hybrid estimator combining BT-transformed offline
  rewards with online pairwise feedback, then apply UCB with a pessimistic bias term.
- **Regret**: O(Σ (Δi+Δj)/max{Δi²,Δj²} · log T − Saving)

### HybElimUCB-RA  (Algorithm 2, Section 5)
- **Setting**: Online stochastic bandits + offline relative (pairwise) data
- **Key idea**: Hybrid UCB on preference probabilities; elimination of clearly
  suboptimal arms using both online-derived and offline-informed confidence sets.
- **Regret**: O(Σ log T/Δi − 2·Ni,1·max{Δi − 2ωi,1, 0})

---

## Experiment Settings (from Appendix G)

| Experiment              | K         | T      | n_trials | Δ    | bias  | N_offline |
|-------------------------|-----------|--------|----------|------|-------|-----------|
| HybUCB-AR scalability   | 8,16,24,32| 50,000 | 100      | 0.1  | 0.1   | 500       |
| HybUCB-AR sensitivity   | 20        | 30,000 | 100      | var  | var   | var       |
| HybElimUCB-RA scalability| 8,16,24,32| 30,000 | 100     | 0.1  | 0.01  | 500       |
| HybElimUCB-RA sensitivity| 10       | 25,000 | 100      | var  | var   | var       |
| Real-world (MovieLens)  | 10        | 10,000 | 50       | —    | —     | 100,000   |
| Real-world (Yelp)       | 10        | 10,000 | 50       | —    | —     | 100,000   |

---

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`

No additional packages required.
