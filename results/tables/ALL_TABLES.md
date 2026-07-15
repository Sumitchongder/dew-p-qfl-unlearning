# QFL-EWP -- Q1 Paper Tables (v3, journal-styled figures; 3 seeds)

## table01_dataset_summary

| Parameter                            | Value                                      |
|:-------------------------------------|:-------------------------------------------|
| Number of clients                    | 5                                          |
| Samples per client                   | 180                                        |
| Train / test split                   | 75% / 25%                                  |
| Feature dimensionality               | 4                                          |
| Non-IID strength (client bias scale) | 0.9                                        |
| Label noise (logistic std)           | 0.2                                        |
| Forgotten client ID                  | 0                                          |
| Task                                 | Binary supply-chain risk classification    |
| Data generation                      | Synthetic, seeded (seed = 1000 + run seed) |


## table02_federated_training_configuration

| Parameter                     | Value                                                      |
|:------------------------------|:-----------------------------------------------------------|
| Ansatz                        | Data re-uploading VQC (RX encode -> RY -> ring-CNOT -> RZ) |
| Qubits                        | 4                                                          |
| Layers                        | 3                                                          |
| Trainable parameters          | 24                                                         |
| Feature encoding scale (rad)  | 0.5                                                        |
| Federated rounds              | 6                                                          |
| Local optimizer               | L-BFGS-B (exact parameter-shift gradient)                  |
| Local iterations / round      | 16                                                         |
| Aggregation rule              | FedAvg, weighted by client sample count                    |
| Gradient estimator            | Exact parameter-shift rule (Pauli generators)              |
| Random seeds                  | 0, 1, 2                                                    |
| Prune fraction (tau, default) | 0.2                                                        |


## table03_baseline_comparison

| method                    |   Accuracy_mean |   Accuracy_std |   AUROC_mean |   AUROC_std |   Forgetting_mean |   Forgetting_std |   MembershipAdv_mean |   MembershipAdv_std |   RetrainDist_mean |   RetrainDist_std |
|:--------------------------|----------------:|---------------:|-------------:|------------:|------------------:|-----------------:|---------------------:|--------------------:|-------------------:|------------------:|
| Random Pruning            |        0.474074 |      0.0401899 |     0.363648 |   0.0559191 |          0.742058 |        0.170696  |           0.257942   |           0.170696  |            6.83151 |          1.43845  |
| Fisher-only Pruning       |        0.466667 |      0.0419435 |     0.4041   |   0.0999829 |          0.896936 |        0.156669  |           0.103064   |           0.156669  |            5.06299 |          1.31003  |
| Entanglement-only Pruning |        0.398148 |      0.0231296 |     0.377947 |   0.066465  |          0.856333 |        0.078515  |           0.0293553  |           0.189294  |            6.23614 |          0.86474  |
| Fine-Tune Only            |        0.864815 |      0.0611952 |     0.961301 |   0.0178573 |          0.825716 |        0.110064  |          -0.00140832 |           0.240154  |            4.21186 |          3.04442  |
| Full Retrain (oracle)     |        0.794444 |      0.030932  |     0.906745 |   0.0350156 |          0.795903 |        0.0388174 |          -0.204097   |           0.0388174 |            0       |          0        |
| QFL-EWP                   |        0.837037 |      0.0285089 |     0.90707  |   0.0336997 |          0.678482 |        0.130007  |          -0.321518   |           0.130007  |            5.0898  |          0.458267 |


## table04_utility_comparison

| Method                    |   Accuracy (mean) |   Accuracy (std) |   AUROC (mean) |   AUROC (std) |   Accuracy_seed0 |   Accuracy_seed1 |   Accuracy_seed2 |
|:--------------------------|------------------:|-----------------:|---------------:|--------------:|-----------------:|-----------------:|-----------------:|
| Random Pruning            |          0.474074 |        0.0401899 |       0.363648 |     0.0559191 |         0.5      |         0.494444 |         0.427778 |
| Fisher-only Pruning       |          0.466667 |        0.0419435 |       0.4041   |     0.0999829 |         0.461111 |         0.427778 |         0.511111 |
| Entanglement-only Pruning |          0.398148 |        0.0231296 |       0.377947 |     0.066465  |         0.405556 |         0.372222 |         0.416667 |
| Fine-Tune Only            |          0.864815 |        0.0611952 |       0.961301 |     0.0178573 |         0.894444 |         0.794444 |         0.905556 |
| Full Retrain (oracle)     |          0.794444 |        0.030932  |       0.906745 |     0.0350156 |         0.822222 |         0.8      |         0.761111 |
| QFL-EWP                   |          0.837037 |        0.0285089 |       0.90707  |     0.0336997 |         0.844444 |         0.805556 |         0.861111 |


## table05_forgetting_comparison

| Method                    |   Forgetting score (mean) |   Forgetting score (std) |   Membership advantage (mean) |   Membership advantage (std) |
|:--------------------------|--------------------------:|-------------------------:|------------------------------:|-----------------------------:|
| Random Pruning            |                  0.742058 |                0.170696  |                    0.257942   |                    0.170696  |
| Fisher-only Pruning       |                  0.896936 |                0.156669  |                    0.103064   |                    0.156669  |
| Entanglement-only Pruning |                  0.856333 |                0.078515  |                    0.0293553  |                    0.189294  |
| Fine-Tune Only            |                  0.825716 |                0.110064  |                   -0.00140832 |                    0.240154  |
| Full Retrain (oracle)     |                  0.795903 |                0.0388174 |                   -0.204097   |                    0.0388174 |
| QFL-EWP                   |                  0.678482 |                0.130007  |                   -0.321518   |                    0.130007  |


## table06_privacy_comparison

| Method                    |   Attack AUC (mean) |   Attack AUC (std) |   Membership advantage (mean) |   Membership advantage (std) | Privacy leakage risk   |
|:--------------------------|--------------------:|-------------------:|------------------------------:|-----------------------------:|:-----------------------|
| Random Pruning            |            0.628971 |          0.0853479 |                    0.257942   |                    0.170696  | High                   |
| Fisher-only Pruning       |            0.551532 |          0.0783346 |                    0.103064   |                    0.156669  | Moderate               |
| Entanglement-only Pruning |            0.514678 |          0.0946472 |                    0.0293553  |                    0.189294  | Low                    |
| Fine-Tune Only            |            0.499296 |          0.120077  |                   -0.00140832 |                    0.240154  | Low                    |
| Full Retrain (oracle)     |            0.397952 |          0.0194087 |                   -0.204097   |                    0.0388174 | Moderate               |
| QFL-EWP                   |            0.339241 |          0.0650034 |                   -0.321518   |                    0.130007  | High                   |


## table07_runtime_comparison

| Method                    |   Unlearning time (s, mean) |   Unlearning time (s, std) |   Speedup vs. Full Retrain |
|:--------------------------|----------------------------:|---------------------------:|---------------------------:|
| Random Pruning            |                 0.000189905 |                1.66295e-05 |           342428           |
| Fisher-only Pruning       |                 5.38403e-05 |                9.971e-06   |                1.20781e+06 |
| Entanglement-only Pruning |                 5.4034e-05  |                2.04431e-05 |                1.20348e+06 |
| Fine-Tune Only            |                14.0326      |                0.179821    |                4.63411     |
| Full Retrain (oracle)     |                65.0287      |                1.16723     |                1           |
| QFL-EWP                   |                 3.96201     |                0.10906     |               16.4131      |


## table08_statistical_significance

| Comparison                            | Metric           |   Mean diff (EWP - baseline) |   t-statistic |   p-value (paired t-test, n=3) | Significant at alpha=0.05   |
|:--------------------------------------|:-----------------|-----------------------------:|--------------:|-------------------------------:|:----------------------------|
| QFL-EWP vs. Random Pruning            | accuracy         |                    0.362963  |       9.95039 |                     0.00994948 | True                        |
| QFL-EWP vs. Random Pruning            | forgetting_score |                   -0.0635757 |      -0.55123 |                     0.636834   | False                       |
| QFL-EWP vs. Fisher-only Pruning       | accuracy         |                    0.37037   |      35.9211  |                     0.0007741  | True                        |
| QFL-EWP vs. Fisher-only Pruning       | forgetting_score |                   -0.218455  |      -5.72944 |                     0.0291383  | True                        |
| QFL-EWP vs. Entanglement-only Pruning | accuracy         |                    0.438889  |     136.832   |                     5.3406e-05 | True                        |
| QFL-EWP vs. Entanglement-only Pruning | forgetting_score |                   -0.177851  |      -5.03083 |                     0.0373138  | True                        |
| QFL-EWP vs. Fine-Tune Only            | accuracy         |                   -0.0277778 |      -1.42374 |                     0.290524   | False                       |
| QFL-EWP vs. Fine-Tune Only            | forgetting_score |                   -0.147234  |      -1.74712 |                     0.222729   | False                       |
| QFL-EWP vs. Full Retrain (oracle)     | accuracy         |                    0.0425926 |       1.46345 |                     0.280899   | False                       |
| QFL-EWP vs. Full Retrain (oracle)     | forgetting_score |                   -0.117421  |      -1.25475 |                     0.336325   | False                       |


## table09_ablation_study

| ablation_type     |   level |   accuracy |    auroc |   forgetting_score |
|:------------------|--------:|-----------:|---------:|-------------------:|
| Non-IID strength  |     0.2 |   0.454545 | 0.566952 |           0.938463 |
| Non-IID strength  |     0.9 |   0.704545 | 0.730619 |           0.702519 |
| Number of clients |     3   |   0.757576 | 0.937327 |           0.903581 |
| Number of clients |     5   |   0.575758 | 0.716452 |           0.770326 |
| Number of clients |     7   |   0.777778 | 0.893722 |           0.94484  |


## table10_hyperparameter_sensitivity

| method   |   Pruning threshold (tau, fraction pruned) |   accuracy |    auroc |   forgetting_score |   retrain_distance |
|:---------|-------------------------------------------:|-----------:|---------:|-------------------:|-------------------:|
| QFL-EWP  |                                        0   |   0.9      | 0.968342 |           0.593909 |            5.37289 |
| QFL-EWP  |                                        0.1 |   0.394444 | 0.352888 |           0.761591 |            5.22582 |
| QFL-EWP  |                                        0.2 |   0.344444 | 0.289443 |           0.922579 |            5.37675 |
| QFL-EWP  |                                        0.3 |   0.411111 | 0.334281 |           0.937503 |            5.66305 |
| QFL-EWP  |                                        0.4 |   0.427778 | 0.409614 |           0.882414 |            7.1692  |
| QFL-EWP  |                                        0.5 |   0.416667 | 0.402248 |           0.595995 |            8.02495 |
| QFL-EWP  |                                        0.6 |   0.5      | 0.447603 |           0.456077 |            7.88379 |

