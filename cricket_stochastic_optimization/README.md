# Smart Cricket XI — Stochastic Optimization

A Streamlit mini project for Stochastic Decision Making. **All included players and performance statistics are synthetic.** This is a teaching simulation, not a real match predictor.

## Setup (Windows / VS Code)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Method
Each player has a simulated batting run outcome (nonnegative capped normal), and eligible bowlers have simulated wickets (Poisson). A documented synthetic utility combines batting, wickets and economy. We optimize exactly 11 players with role and availability constraints. The deterministic baseline maximizes sample expected utility. The stochastic version maximizes `(1-λ) E[utility] + λ CVaR_10%(utility)` using a mixed-integer linear program (PuLP/CBC). Both methods are compared on 5,000 fresh scenarios with a different random seed. Risk λ=0 matches the baseline. Risk λ=1 maximizes the lower-tail mean. The model assumes independent players and does not account for batting-order interactions or opposition.

## Files
- `app.py`: seven-page interactive dashboard
- `modules/core.py`: validation, simulation, optimization, evaluation
- `data/cricket_players.csv`: synthetic dataset with 46 players

## Notes
The training slider allows up to 10,000 simulations for individual player analysis; the optimization uses at most 1,000 training scenarios to keep CBC responsive. Sensitivity analysis uses 250 training scenarios. Model utility scores are **not** literal team runs. Uploaded CSV must have the same columns as the included dataset. Actual improvements are not guaranteed. The project is an educational prototype, not a validated sports decision system.
