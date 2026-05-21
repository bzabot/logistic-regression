# Add-on Performance Analysis

- Completed runs: 6160
- Datasets: 20
- Seeds: 20
- Models: 16

## Top Models By Mean Balanced Accuracy

| model | balanced_accuracy_mean | minority_f1_mean | minority_recall_mean |
| --- | --- | --- | --- |
| cagd_f1_threshold | 0.786011 | 0.597594 | 0.700024 |
| gaar_cagd_f1_threshold | 0.783624 | 0.599603 | 0.705909 |
| gaar_cagd | 0.783201 | 0.554069 | 0.602621 |
| gaar_focal_loss_cagd_f1_threshold | 0.77335 | 0.589032 | 0.697026 |
| focal_loss_cagd_f1_threshold | 0.772822 | 0.578211 | 0.692104 |
| gaar_f1_threshold | 0.767923 | 0.574999 | 0.675884 |
| f1_threshold | 0.760177 | 0.555458 | 0.662526 |
| gaar_focal_loss_f1_threshold | 0.759937 | 0.555047 | 0.674616 |
| gaar_focal_loss_cagd | 0.752632 | 0.457832 | 0.56692 |
| cagd | 0.750541 | 0.533538 | 0.523557 |

## Balanced Accuracy Tests Vs Default

| model | n_pairs | mean_difference | bootstrap_ci_low | bootstrap_ci_high | wins | ties | losses | permutation_p_value | holm_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cagd_f1_threshold | 385 | 0.124278 | 0.108296 | 0.139698 | 296 | 19 | 70 | 9.999e-05 | 0.00149985 |
| gaar_cagd_f1_threshold | 385 | 0.12189 | 0.104551 | 0.138019 | 296 | 24 | 65 | 9.999e-05 | 0.00149985 |
| gaar_cagd | 385 | 0.121468 | 0.108125 | 0.13479 | 270 | 70 | 45 | 9.999e-05 | 0.00149985 |
| gaar_focal_loss_cagd_f1_threshold | 385 | 0.111617 | 0.0951545 | 0.127303 | 291 | 26 | 68 | 9.999e-05 | 0.00149985 |
| focal_loss_cagd_f1_threshold | 385 | 0.111088 | 0.0959158 | 0.12619 | 284 | 26 | 75 | 9.999e-05 | 0.00149985 |
| gaar_f1_threshold | 385 | 0.106189 | 0.0921603 | 0.120185 | 313 | 23 | 49 | 9.999e-05 | 0.00149985 |
| f1_threshold | 385 | 0.0984429 | 0.0847526 | 0.111803 | 294 | 26 | 65 | 9.999e-05 | 0.00149985 |
| gaar_focal_loss_f1_threshold | 385 | 0.0982037 | 0.0845325 | 0.11164 | 294 | 26 | 65 | 9.999e-05 | 0.00149985 |
| gaar_focal_loss_cagd | 385 | 0.0908984 | 0.0790699 | 0.102781 | 229 | 73 | 83 | 9.999e-05 | 0.00149985 |
| cagd | 385 | 0.0888072 | 0.076839 | 0.101006 | 229 | 106 | 50 | 9.999e-05 | 0.00149985 |
| focal_loss_f1_threshold | 385 | 0.0867019 | 0.0730338 | 0.100128 | 263 | 25 | 97 | 9.999e-05 | 0.00149985 |
| focal_loss_cagd | 385 | 0.0619951 | 0.0529266 | 0.0714427 | 219 | 92 | 74 | 9.999e-05 | 0.00149985 |
| gaar | 385 | 0.0605521 | 0.0520965 | 0.0692293 | 204 | 137 | 44 | 9.999e-05 | 0.00149985 |
| gaar_focal_loss | 385 | 0.0513964 | 0.0427335 | 0.0603798 | 168 | 134 | 83 | 9.999e-05 | 0.00149985 |
| focal_loss | 385 | -0.00529515 | -0.0100373 | -0.000617065 | 48 | 201 | 136 | 0.0307969 | 0.0307969 |