# wine_classification

Different approaches to classify wine based on their chemical composition.

## Dataset

The data in `example_data/` is the classic [Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine), the result of a chemical analysis of wines grown in the same region of Italy but derived from three different cultivars.

- **Instances:** 178 wines (class 1: 59, class 2: 71, class 3: 48)
- **Attributes:** 13 continuous chemical measurements, plus a class label (1–3) identifying the cultivar
- **Missing values:** none

Attributes:

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

Files:

- `example_data/wine.data` — the raw data, one wine per row, first column is the class label
- `example_data/wine.names` — full dataset documentation
- `example_data/Index` — UCI repository index file

## Structure

- `python/` — Python approaches (pandas, scikit-learn, etc.)
- `matlab/` — MATLAB approaches

## License

See [LICENSE](LICENSE).
