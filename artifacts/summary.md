# temporal_pattern_biomarker_benchmark

Modo: `synthetic`

## Melhor modelo

- Modelo: **logistic_regression**
- F1 macro: **0.9421**
- Accuracy: **0.9422**
- Log loss: **0.1589**
- Brier score: **0.0294**

## Tabela comparativa

| model               |   accuracy |   f1_macro |   log_loss |   brier_score |   fit_time_sec |   predict_time_sec_per_sample |
|:--------------------|-----------:|-----------:|-----------:|--------------:|---------------:|------------------------------:|
| logistic_regression |   0.942222 |   0.942148 |   0.15891  |     0.0294179 |       0.256557 |                   4.86888e-06 |
| random_forest       |   0.931111 |   0.931101 |   0.189874 |     0.0347211 |       5.69041  |                   0.00185689  |
| mlp                 |   0.931111 |   0.930778 |   0.222569 |     0.038942  |       0.437729 |                   5.9783e-06  |
| gradient_boosting   |   0.925556 |   0.92563  |   0.225696 |     0.0388185 |       5.41534  |                   1.79686e-05 |

## Interpretação sugerida

Modelos com maior F1 macro e menor log loss indicam melhor separação dos biomarcadores entre estímulos. Quando o modo for `replay` ou `sdk`, diferenças para o benchmark sintético podem indicar limitações do simulador ou maior variabilidade biológica.
