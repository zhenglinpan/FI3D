# FI3D: Revisiting Frechet Inception Distance for 3D Human Motion Generation

<p align="center">
  <img src="https://github.com/zhenglinpan/FI3D/blob/master/assets/page.png">
</p>
<p align="center">
  <img src="https://github.com/zhenglinpan/FI3D/blob/master/assets/exp.png">
</p>

****

**Welcome to the FI3D repository!**

Our project aims to enhance the evaluation of 3D human motion generation models. Traditional methods like Frechet Inception Distance (FID) assume Gaussian data distribution, which often leads to inaccurate results. 

**Authors**: [Zhenglin Pan¹](https://github.com/zhenglinpan), [Mahyar Karami¹](https://github.com/alivosoughi), [Alireza Vosoughi Rad¹](https://github.com/Mahyar-Karami)

**Affiliation**: University of Alberta

**Abstract**

We introduce FI3D, a novel evaluation metric that overcomes FID's limitations by using Gaussian Mixture Models (GMMs). Our experiments on the HumanML3D dataset show that FI3D offers more stable and accurate evaluations of generated human motion sequences.

Explore our code and resources to learn more about FI3D and how it improves 3D human motion assessment.

## How to re-implement
### Step 0

We conducted experiments with several recent researches on 3D human motion generation, including [action2motion](https://github.com/EricGuo5513/action-to-motion), [ACTOR](https://github.com/Mathux/ACTOR), [MDM](https://github.com/GuyTevet/motion-diffusion-model),[MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse), [priorMDM](https://github.com/priorMDM/priorMDM), [Text2Motion](https://github.com/EricGuo5513/text-to-motion). Re-implementing our experiments requires setting up the environment and compiling the dataset for each of the models, while this could be a tedious process, we provide the following results in jupyter notebook in under `gaussian_tests` folder to help you get fast results. 

Our research mainly reuses the code implementations from [Text2Motion](https://github.com/EricGuo5513/text-to-motion) where FID was used for evaluating 3D human motion, please follow the repo's instructions to set up the environment and compile the dataset properly untill you can run the code.

After everything's set up, clone this repo, copy&overwrite the files in the repo to the corresponding folders in the Text2Motion folder.

### Step 1
Save the ground truth motions and generated motions to the corresponding folders by running:
```
python save_motions.py
```
The result will be saved to `./eval_data` folder.

### Step 2
With the saved motions, run the following command to degenerate motions with controlled perturbation:
```python
python eval_motions.py
```
The result will be saved to `./data_fi3d/emb` and `./data_fi3d/joint` folders, where the `emb` folder contains the degenerated motions in the embedding space and the `joint` folder contains the degenerated motions in the joint space.


### Step 3(optional)
Run the following command to visualize the degenerated motions, this step is optional and only for visualization purpose.
```python
python animate.py
```
The result will be saved to `./data_fi3d/animations` folder.

### Step 4
Run `eval.ipynb`
Follow the instructions in the notebook to evaluate the generated motions with FI3D.
Run the notebook cell by cell, the result will be saved to `.eval_results.csv` and `eval_results_time.csv`. 