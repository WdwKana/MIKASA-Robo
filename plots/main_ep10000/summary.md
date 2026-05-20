# Main Experiments — last-3-eval summary (ep10000)

tasks: 6, seeds: ['33', '42', '99']
Per §6.2: average of last 3 eval checkpoints per seed; mean±std across seeds.

## success_once

| task                           | method               | n | mean±std        | per_seed |
|--------------------------------|----------------------|---|-----------------|----------------------------------------|
| RememberColor9-v0              | CVAE-v6 (Ours)       | 3 | 0.219±0.029     | s33:0.203, s42:0.194, s99:0.260 |
| RememberColor9-v0              | PPO+Believer         | 3 | 0.225±0.034     | s33:0.205, s42:0.273, s99:0.198 |
| RememberColor9-v0              | PPO                  | 3 | 0.067±0.095     | s33:0.201, s42:0.000, s99:0.000 |
| RememberColor9-v0              | PPO+LSTM             | 3 | 0.117±0.042     | s33:0.134, s42:0.158, s99:0.059 |
| RememberColor9-v0              | PPO+GRU              | 3 | 0.042±0.045     | s33:0.003, s42:0.106, s99:0.017 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x2-v0    | CVAE-v6 (Ours)       | 3 | 0.674±0.109     | s33:0.538, s42:0.806, s99:0.677 |
| RememberShapeAndColor3x2-v0    | PPO+Believer         | 3 | 0.700±0.051     | s33:0.670, s42:0.658, s99:0.773 |
| RememberShapeAndColor3x2-v0    | PPO                  | 3 | 0.000±0.000     | s33:0.000, s42:0.000, s99:0.000 |
| RememberShapeAndColor3x2-v0    | PPO+LSTM             | 3 | 0.220±0.045     | s33:0.271, s42:0.161, s99:0.229 |
| RememberShapeAndColor3x2-v0    | PPO+GRU              | 3 | 0.003±0.002     | s33:0.000, s42:0.003, s99:0.005 |
|                                |                      |   |                 |                                        |
| RememberShape9-v0              | CVAE-v6 (Ours)       | 3 | 0.060±0.019     | s33:0.061, s42:0.083, s99:0.036 |
| RememberShape9-v0              | PPO+Believer         | 3 | 0.102±0.035     | s33:0.151, s42:0.082, s99:0.073 |
| RememberShape9-v0              | PPO                  | 3 | 0.055±0.040     | s33:0.071, s42:0.094, s99:0.000 |
| RememberShape9-v0              | PPO+LSTM             | 3 | 0.161±0.024     | s33:0.189, s42:0.163, s99:0.130 |
| RememberShape9-v0              | PPO+GRU              | 3 | 0.003±0.004     | s33:0.000, s42:0.009, s99:0.000 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x3-v0    | CVAE-v6 (Ours)       | 3 | 0.199±0.048     | s33:0.207, s42:0.137, s99:0.253 |
| RememberShapeAndColor3x3-v0    | PPO+Believer         | 3 | 0.207±0.034     | s33:0.215, s42:0.243, s99:0.161 |
| RememberShapeAndColor3x3-v0    | PPO                  | 3 | 0.029±0.041     | s33:0.000, s42:0.087, s99:0.000 |
| RememberShapeAndColor3x3-v0    | PPO+LSTM             | 3 | 0.170±0.054     | s33:0.247, s42:0.132, s99:0.130 |
| RememberShapeAndColor3x3-v0    | PPO+GRU              | 3 | 0.017±0.015     | s33:0.014, s42:0.000, s99:0.036 |
|                                |                      |   |                 |                                        |
| InterceptFast-v0               | CVAE-v6 (Ours)       | 3 | 0.635±0.065     | s33:0.722, s42:0.566, s99:0.616 |
| InterceptFast-v0               | PPO+Believer         | 3 | 0.633±0.114     | s33:0.790, s42:0.589, s99:0.521 |
| InterceptFast-v0               | PPO                  | 3 | 0.189±0.268     | s33:0.000, s42:0.568, s99:0.000 |
| InterceptFast-v0               | PPO+LSTM             | 3 | 0.575±0.018     | s33:0.562, s42:0.601, s99:0.563 |
| InterceptFast-v0               | PPO+GRU              | 3 | 0.178±0.251     | s33:0.000, s42:0.533, s99:0.000 |
|                                |                      |   |                 |                                        |
| InterceptMedium-v0             | CVAE-v6 (Ours)       | 3 | 0.593±0.058     | s33:0.531, s42:0.576, s99:0.670 |
| InterceptMedium-v0             | PPO+Believer         | 3 | 0.673±0.141     | s33:0.484, s42:0.712, s99:0.823 |
| InterceptMedium-v0             | PPO                  | 3 | 0.705±0.091     | s33:0.632, s42:0.833, s99:0.649 |
| InterceptMedium-v0             | PPO+LSTM             | 3 | 0.473±0.029     | s33:0.441, s42:0.510, s99:0.469 |
| InterceptMedium-v0             | PPO+GRU              | 3 | 0.031±0.022     | s33:0.026, s42:0.061, s99:0.007 |
|                                |                      |   |                 |                                        |

## success_at_end

| task                           | method               | n | mean±std        | per_seed |
|--------------------------------|----------------------|---|-----------------|----------------------------------------|
| RememberColor9-v0              | CVAE-v6 (Ours)       | 3 | 0.130±0.014     | s33:0.135, s42:0.111, s99:0.144 |
| RememberColor9-v0              | PPO+Believer         | 3 | 0.144±0.021     | s33:0.141, s42:0.172, s99:0.120 |
| RememberColor9-v0              | PPO                  | 3 | 0.032±0.045     | s33:0.095, s42:0.000, s99:0.000 |
| RememberColor9-v0              | PPO+LSTM             | 3 | 0.077±0.039     | s33:0.083, s42:0.122, s99:0.026 |
| RememberColor9-v0              | PPO+GRU              | 3 | 0.028±0.034     | s33:0.003, s42:0.076, s99:0.003 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x2-v0    | CVAE-v6 (Ours)       | 3 | 0.526±0.132     | s33:0.387, s42:0.703, s99:0.488 |
| RememberShapeAndColor3x2-v0    | PPO+Believer         | 3 | 0.626±0.063     | s33:0.569, s42:0.595, s99:0.714 |
| RememberShapeAndColor3x2-v0    | PPO                  | 3 | 0.000±0.000     | s33:0.000, s42:0.000, s99:0.000 |
| RememberShapeAndColor3x2-v0    | PPO+LSTM             | 3 | 0.073±0.024     | s33:0.087, s42:0.094, s99:0.040 |
| RememberShapeAndColor3x2-v0    | PPO+GRU              | 3 | 0.001±0.001     | s33:0.000, s42:0.000, s99:0.002 |
|                                |                      |   |                 |                                        |
| RememberShape9-v0              | CVAE-v6 (Ours)       | 3 | 0.031±0.007     | s33:0.028, s42:0.040, s99:0.024 |
| RememberShape9-v0              | PPO+Believer         | 3 | 0.040±0.002     | s33:0.043, s42:0.038, s99:0.038 |
| RememberShape9-v0              | PPO                  | 3 | 0.042±0.033     | s33:0.047, s42:0.080, s99:0.000 |
| RememberShape9-v0              | PPO+LSTM             | 3 | 0.086±0.013     | s33:0.085, s42:0.102, s99:0.069 |
| RememberShape9-v0              | PPO+GRU              | 3 | 0.002±0.003     | s33:0.000, s42:0.007, s99:0.000 |
|                                |                      |   |                 |                                        |
| RememberShapeAndColor3x3-v0    | CVAE-v6 (Ours)       | 3 | 0.111±0.038     | s33:0.111, s42:0.064, s99:0.156 |
| RememberShapeAndColor3x3-v0    | PPO+Believer         | 3 | 0.127±0.027     | s33:0.116, s42:0.165, s99:0.101 |
| RememberShapeAndColor3x3-v0    | PPO                  | 3 | 0.024±0.034     | s33:0.000, s42:0.071, s99:0.000 |
| RememberShapeAndColor3x3-v0    | PPO+LSTM             | 3 | 0.050±0.050     | s33:0.118, s42:0.002, s99:0.029 |
| RememberShapeAndColor3x3-v0    | PPO+GRU              | 3 | 0.003±0.003     | s33:0.007, s42:0.000, s99:0.002 |
|                                |                      |   |                 |                                        |
| InterceptFast-v0               | CVAE-v6 (Ours)       | 3 | 0.084±0.016     | s33:0.078, s42:0.106, s99:0.068 |
| InterceptFast-v0               | PPO+Believer         | 3 | 0.096±0.016     | s33:0.118, s42:0.089, s99:0.082 |
| InterceptFast-v0               | PPO                  | 3 | 0.010±0.014     | s33:0.000, s42:0.029, s99:0.000 |
| InterceptFast-v0               | PPO+LSTM             | 3 | 0.035±0.014     | s33:0.019, s42:0.031, s99:0.054 |
| InterceptFast-v0               | PPO+GRU              | 3 | 0.013±0.019     | s33:0.000, s42:0.040, s99:0.000 |
|                                |                      |   |                 |                                        |
| InterceptMedium-v0             | CVAE-v6 (Ours)       | 3 | 0.124±0.021     | s33:0.132, s42:0.144, s99:0.095 |
| InterceptMedium-v0             | PPO+Believer         | 3 | 0.122±0.054     | s33:0.196, s42:0.068, s99:0.101 |
| InterceptMedium-v0             | PPO                  | 3 | 0.025±0.013     | s33:0.009, s42:0.042, s99:0.026 |
| InterceptMedium-v0             | PPO+LSTM             | 3 | 0.041±0.013     | s33:0.035, s42:0.059, s99:0.029 |
| InterceptMedium-v0             | PPO+GRU              | 3 | 0.002±0.000     | s33:0.002, s42:0.002, s99:0.002 |
|                                |                      |   |                 |                                        |

