# McLatte
![Pytest](https://github.com/Jason-Y-Z/McLatte/actions/workflows/python-package.yml/badge.svg)

Multi-Cause LAtent facTor iTe Estimator

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install mclatte.

```bash
pip install -r requirements.txt
pip install .
```

## Usage

An example workflow with McLatte and idealised disease treatment is illustrated below

```python
from mclatte.mclatte.model import (
    infer_mcespresso,
    train_mclatte,
)
from mclatte.mclatte.simulation_data import (
    generate_data, 
    TreatmentRepr,
    SimDataGenConfig
)

M = 5
H = 5
R = 5
D = 10
K = 3
C = 4
constants = dict(m=M, h=H, r=R, d=D, k=K, c=C)
data_gen_config = SimDataGenConfig(
    n=200, p_0=0.1, mode=TreatmentRepr.BINARY, **constants
)


# Generate data
(
    _,
    train_data,
    test_data,
) = generate_data(data_gen_config, return_raw=False)

# Example McLatte configurations
mclatte_config = {
    "encoder_class": "lstm",
    "decoder_class": "lstm",
    "hidden_dim": 16,
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.024468,
    "gamma": 0.740409,
    "lambda_r": 0.040299,
    "lambda_d": 0.034368,
    "lambda_p": 0.021351,
}

# Training
trained_mclatte = train_mclatte(
    mclatte_config,
    constants,
    train_data,
)

# Inference
_, _, y_tilde = infer_mcespresso(
    trained_mclatte, test_data['x'], test_data['a'], test_data['t'], test_data['m']
)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause License](https://github.com/Jason-Y-Z/McLatte/blob/main/LICENSE)