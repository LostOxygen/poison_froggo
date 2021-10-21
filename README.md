# Poison_Froggo
re-implementation of the poison frogs! paper in PyTorch by Shafahi et al.:
* [Arxiv](https://arxiv.org/abs/1804.00792)
* [Github](https://github.com/ashafahi/inceptionv3-transferLearn-poison)

## Run the attack
Create **test** and **train** folder as subfolders under **./poison_frog/datasets/** and place **dog** and **fish** named folders with train and test data inside. The original paper uses around 900 training images each and 698 test images for dog class and 401 test images for fish class.
The imagenet data can be found [here](https://www.kaggle.com/c/imagenet-object-localization-challenge).

After preparing the data simply run
```python3
python run_frog.py
```
for a single attack with only one poison, or
```python3
python run_frog.py --num_poisons X --num_runs Y
```
for X poisons and Y attack runs. The attack will be evaluated when finished and will print the successrate depending on the total attack runs.
