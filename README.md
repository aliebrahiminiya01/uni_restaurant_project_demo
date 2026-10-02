# University Restaurant: Meal No-Show Prediction

Students reserve their university restaurant meals in advance, but about **15%** of reserved meals are never
picked up, and every no-show is wasted food. This project predicts which reservations are likely to be
**no-shows**, so the kitchen can plan how much to cook.

## Data

One semester (15 Feb – 20 Jun 2025) of reservations from 1,000 students — 52,333 reservations in total.

| File | Contents |
|------|----------|
| `reservations.csv` | One row per meal reservation; `picked_up` is the target (1 = picked up, 0 = no-show) |
| `student_profiles.csv` | Whether each student lives in a dormitory |
| `weekly_menu.csv` | The weekly menu (meal per weekday and meal type) |
| `academic_calendar_daily.csv` | Teaching, exam or break period for every day |
| `national_calendar.csv` | National holidays |

## Approach

- **No data leakage:** `pickup_time` is only known after a meal is picked up, so it is not used as a feature.
- **Student history feature:** each reservation gets the student's smoothed pickup rate computed **only from earlier days**.
- **Time-based split:** trained on the first 80% of the semester, tested on the last 20% (future days).
- **One-hot encoding** for categorical features and `class_weight='balanced'` for the rare no-show class.
- Models are compared with **ROC AUC** and **average precision for no-shows**, because accuracy is misleading:
  predicting "everyone picks up" is already ~85% accurate.

## Results (test period: 27 May – 20 Jun 2025)

| Model | ROC AUC | Avg. precision (no-show) |
|-------|---------|--------------------------|
| Baseline (everyone picks up) | 0.500 | 0.146 |
| Logistic regression, no student history | 0.554 | 0.170 |
| **Logistic regression + student history** | **0.663** | **0.279** |
| Gradient boosting + student history | 0.656 | 0.265 |

## Key findings

1. About **14.5%** of reserved meals are never picked up.
2. **Holidays** are the biggest risk: more than half of the meals reserved for Nowruz, Labor Day and the Spring Festival are wasted. **Breakfast** has more no-shows than lunch and dinner.
3. A student's **own past pickup rate** is the most useful predictor. Without it the model is close to random (ROC AUC 0.55); with it, ROC AUC rises to **0.66** and average precision for no-shows to **0.28** (a random guess scores 0.15).
4. Logistic regression with one-hot encoded features performs as well as gradient boosting and is easier to interpret.
5. At a threshold of 0.5 the model flags 20% of reservations and catches about 39% of the no-shows. A stricter threshold (0.6) flags only 10% of reservations, with 39% of the flagged ones being real no-shows.

**Possible next steps:** add holiday and pre-holiday periods to the test set (e.g. a second semester), use features such as days between reservation and meal, and estimate how many portions the kitchen could save per day.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook no_show_prediction.ipynb
```
