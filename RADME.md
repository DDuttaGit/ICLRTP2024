# Lost in Translation: GANs' Inability to Generate Simple Probability Distributions

## Anonymous

### Configuration
The file `config` contains necessary instruction set to run the experiment. The following table describes the structure of the file `config`:

| Parameter | Description | Values it can take|
| ------ | ------ | ------- |
| GAN | A specific model to choose from the list of Vanilla GAN, Wasserstein GAN or Least Square GAN | `VGAN`, `WGAN` or `LSGAN`
| MU | The mean Target Gaussian Distribution that is expected | any float value
| SIGMA | The standard deviation Target Gaussian Distribution that is expected | any float value
| NOISE | The Noise Distribution either an Uniform Distribution within range (0, 1) or a Standard Normal Distribution | `U` or `N`
| EPOCHS | Number of Epochs | any integer

### Starting the Experiment
Once the `config` file is prepared:
```sh
python run.py
```
