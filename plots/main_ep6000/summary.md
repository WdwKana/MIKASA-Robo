# Main Experiments — last-3-eval summary (ep6000)

tasks: 6, seeds: ['33', '42', '99']
Per §6.2: average of last 3 eval checkpoints per seed; mean±std across seeds.

## success_once

| task                           | method               | n | mean±std        | per_seed |
|--------------------------------|----------------------|---|-----------------|----------------------------------------|
| RememberColor9-v0              | CVAE-v6 (Ours)       | 3 | 0.164±0.019     | s33:0.179, s42:0.137, s99:0.177 |
| RememberColor9-v0              | PPO+Believer         | 3 | 0.196±0.092     | s33:0.325, s42:0.115, s99:0.148 |
| RememberColor9-v0              | PPO                  | 3 | 0.067±0.095     | s33:0.201, s42:0.000, s99:0.000 |
| RememberColor9-v0              | PPO+LSTM             | 3 | 0.117±0.042     | s33:0.134, s42:0.158, s99:0.059 |
| RememberColor9-v0              | PPO+GRU              | 3 | 0.042±0.045     | s33:0.003, s42:0.106, s99:0.017 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x2-v0    | CVAE-v6 (Ours)       | 3 | 0.568±0.040     | s33:0.587, s42:0.512, s99:0.606 |
| RememberShapeAndColor3x2-v0    | PPO+Believer         | 3 | 0.626±0.040     | s33:0.585, s42:0.681, s99:0.613 |
| RememberShapeAndColor3x2-v0    | PPO                  | 3 | 0.000±0.000     | s33:0.000, s42:0.000, s99:0.000 |
| RememberShapeAndColor3x2-v0    | PPO+LSTM             | 3 | 0.220±0.045     | s33:0.271, s42:0.161, s99:0.229 |
| RememberShapeAndColor3x2-v0    | PPO+GRU              | 3 | 0.003±0.002     | s33:0.000, s42:0.003, s99:0.005 |
|                                |                      |   |                 |                                        |
| RememberShape9-v0              | CVAE-v6 (Ours)       | 3 | 0.047±0.019     | s33:0.064, s42:0.021, s99:0.056 |
| RememberShape9-v0              | PPO+Believer         | 3 | 0.082±0.050     | s33:0.148, s42:0.069, s99:0.028 |
| RememberShape9-v0              | PPO                  | 3 | 0.055±0.040     | s33:0.071, s42:0.094, s99:0.000 |
| RememberShape9-v0              | PPO+LSTM             | 3 | 0.161±0.024     | s33:0.189, s42:0.163, s99:0.130 |
| RememberShape9-v0              | PPO+GRU              | 3 | 0.003±0.004     | s33:0.000, s42:0.009, s99:0.000 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x3-v0    | CVAE-v6 (Ours)       | 3 | 0.139±0.039     | s33:0.194, s42:0.113, s99:0.109 |
| RememberShapeAndColor3x3-v0    | PPO+Believer         | 3 | 0.184±0.016     | s33:0.203, s42:0.163, s99:0.186 |
| RememberShapeAndColor3x3-v0    | PPO                  | 3 | 0.029±0.041     | s33:0.000, s42:0.087, s99:0.000 |
| RememberShapeAndColor3x3-v0    | PPO+LSTM             | 3 | 0.170±0.054     | s33:0.247, s42:0.132, s99:0.130 |
| RememberShapeAndColor3x3-v0    | PPO+GRU              | 3 | 0.017±0.015     | s33:0.014, s42:0.000, s99:0.036 |
|                                |                      |   |                 |                                        |
| InterceptFast-v0               | CVAE-v6 (Ours)       | 3 | 0.524±0.061     | s33:0.538, s42:0.443, s99:0.590 |
| InterceptFast-v0               | PPO+Believer         | 3 | 0.564±0.084     | s33:0.661, s42:0.457, s99:0.575 |
| InterceptFast-v0               | PPO                  | 3 | 0.189±0.268     | s33:0.000, s42:0.568, s99:0.000 |
| InterceptFast-v0               | PPO+LSTM             | 3 | 0.575±0.018     | s33:0.562, s42:0.601, s99:0.563 |
| InterceptFast-v0               | PPO+GRU              | 3 | 0.178±0.251     | s33:0.000, s42:0.533, s99:0.000 |
|                                |                      |   |                 |                                        |
| InterceptMedium-v0             | CVAE-v6 (Ours)       | 3 | 0.621±0.059     | s33:0.675, s42:0.538, s99:0.649 |
| InterceptMedium-v0             | PPO+Believer         | 3 | 0.535±0.101     | s33:0.398, s42:0.635, s99:0.573 |
| InterceptMedium-v0             | PPO                  | 3 | 0.705±0.091     | s33:0.632, s42:0.833, s99:0.649 |
| InterceptMedium-v0             | PPO+LSTM             | 3 | 0.473±0.029     | s33:0.441, s42:0.510, s99:0.469 |
| InterceptMedium-v0             | PPO+GRU              | 3 | 0.031±0.022     | s33:0.026, s42:0.061, s99:0.007 |
|                                |                      |   |                 |                                        |

## success_at_end

| task                           | method               | n | mean±std        | per_seed |
|--------------------------------|----------------------|---|-----------------|----------------------------------------|
| RememberColor9-v0              | CVAE-v6 (Ours)       | 3 | 0.102±0.022     | s33:0.123, s42:0.071, s99:0.111 |
| RememberColor9-v0              | PPO+Believer         | 3 | 0.128±0.080     | s33:0.240, s42:0.056, s99:0.089 |
| RememberColor9-v0              | PPO                  | 3 | 0.032±0.045     | s33:0.095, s42:0.000, s99:0.000 |
| RememberColor9-v0              | PPO+LSTM             | 3 | 0.077±0.039     | s33:0.083, s42:0.122, s99:0.026 |
| RememberColor9-v0              | PPO+GRU              | 3 | 0.028±0.034     | s33:0.003, s42:0.076, s99:0.003 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x2-v0    | CVAE-v6 (Ours)       | 3 | 0.439±0.037     | s33:0.415, s42:0.411, s99:0.491 |
| RememberShapeAndColor3x2-v0    | PPO+Believer         | 3 | 0.528±0.083     | s33:0.431, s42:0.634, s99:0.519 |
| RememberShapeAndColor3x2-v0    | PPO                  | 3 | 0.000±0.000     | s33:0.000, s42:0.000, s99:0.000 |
| RememberShapeAndColor3x2-v0    | PPO+LSTM             | 3 | 0.073±0.024     | s33:0.087, s42:0.094, s99:0.040 |
| RememberShapeAndColor3x2-v0    | PPO+GRU              | 3 | 0.001±0.001     | s33:0.000, s42:0.000, s99:0.002 |
|                                |                      |   |                 |                                        |
| RememberShape9-v0              | CVAE-v6 (Ours)       | 3 | 0.030±0.016     | s33:0.049, s42:0.010, s99:0.031 |
| RememberShape9-v0              | PPO+Believer         | 3 | 0.046±0.031     | s33:0.085, s42:0.045, s99:0.009 |
| RememberShape9-v0              | PPO                  | 3 | 0.042±0.033     | s33:0.047, s42:0.080, s99:0.000 |
| RememberShape9-v0              | PPO+LSTM             | 3 | 0.086±0.013     | s33:0.085, s42:0.102, s99:0.069 |
| RememberShape9-v0              | PPO+GRU              | 3 | 0.002±0.003     | s33:0.000, s42:0.007, s99:0.000 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x3-v0    | CVAE-v6 (Ours)       | 3 | 0.071±0.035     | s33:0.120, s42:0.054, s99:0.038 |
| RememberShapeAndColor3x3-v0    | PPO+Believer         | 3 | 0.124±0.019     | s33:0.151, s42:0.108, s99:0.115 |
| RememberShapeAndColor3x3-v0    | PPO                  | 3 | 0.024±0.034     | s33:0.000, s42:0.071, s99:0.000 |
| RememberShapeAndColor3x3-v0    | PPO+LSTM             | 3 | 0.050±0.050     | s33:0.118, s42:0.002, s99:0.029 |
| RememberShapeAndColor3x3-v0    | PPO+GRU              | 3 | 0.003±0.003     | s33:0.007, s42:0.000, s99:0.002 |
|                                |                      |   |                 |                                        |
| InterceptFast-v0               | CVAE-v6 (Ours)       | 3 | 0.086±0.013     | s33:0.101, s42:0.069, s99:0.087 |
| InterceptFast-v0               | PPO+Believer         | 3 | 0.076±0.028     | s33:0.068, s42:0.047, s99:0.115 |
| InterceptFast-v0               | PPO                  | 3 | 0.010±0.014     | s33:0.000, s42:0.029, s99:0.000 |
| InterceptFast-v0               | PPO+LSTM             | 3 | 0.035±0.014     | s33:0.019, s42:0.031, s99:0.054 |
| InterceptFast-v0               | PPO+GRU              | 3 | 0.013±0.019     | s33:0.000, s42:0.040, s99:0.000 |
|                                |                      |   |                 |                                        |
| InterceptMedium-v0             | CVAE-v6 (Ours)       | 3 | 0.130±0.023     | s33:0.134, s42:0.156, s99:0.101 |
| InterceptMedium-v0             | PPO+Believer         | 3 | 0.096±0.024     | s33:0.066, s42:0.125, s99:0.097 |
| InterceptMedium-v0             | PPO                  | 3 | 0.025±0.013     | s33:0.009, s42:0.042, s99:0.026 |
| InterceptMedium-v0             | PPO+LSTM             | 3 | 0.041±0.013     | s33:0.035, s42:0.059, s99:0.029 |
| InterceptMedium-v0             | PPO+GRU              | 3 | 0.002±0.000     | s33:0.002, s42:0.002, s99:0.002 |
|                                |                      |   |                 |                                        |

