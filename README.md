# Udacity Machine Learning Engineer on Azure Capstone: Mobile Phone Prices Classification


In this project, we use the knowledge obtained from the mentioned Nanodegree to 
solve an interesting problem: **Mobile Prices Classification Based on Technical Charateristics**. 

In this project, two models will be created: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

This project aims to demonstrate the ability to use an external dataset in our workspace, train a model using the different tools available in the AzureML framework as well as our ability to deploy the model as a web service.

<figure style='text-align:center'>
    <img src='img/capstone-diagram.png' alt='diagram' style="width:50%"/>
    <figcaption style='text-align:center'>Figure 1: Project Workflow</figcaption>
</figure>

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

For this project, the data used is **Mobile Price Classification** ([data source](https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv))
from Kaggle website. The description provided in Kaggle is the following one:

```
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is.
```

We are using the *train.csv* file.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

As described above, we are using some technical characteristics of mobile phones
to classify their prices between 0 and 3. So that, we have a Multi-Label
Classification Problem.

The features available are the following:

* **battery_power**: Total energy a battery can store in one time measured in mAh.

* **blue**: Has bluetooth or not.

* **clock_speed**: speed at which microprocessor executes instructions.

* **dual_sim**: Has dual sim support or not.

* **fc**: Front Camera mega pixels

* **four_g**: Has 4G or not.

* **int_memory**: Internal Memory in Gigabytes.

* **m_dep**: Mobile Depth in cm.

* **mobile_wt**: Weight of mobile phone.

* **n_cores**: Number of cores of processor.

* **pc**: Primary Camera mega pixels.

* **px_height**: Pixel Resolution Height.

* **px_width**: Pixel Resolution Width.

* **ram**: Random Access Memory in Mega Bytes.

* **sc_h**: Screen Height of mobile in cm.

* **sc_w**: Screen Width of mobile in cm.

* **talk_time**: longest time that a single battery charge will last when you are.

* **three_g**: Has 3G or not.

* **touch_screen**: Has touch screen or not.

* **wifi**: Has wifi or not.

* **price_range**: This is the target variable with value of 0 (low cost), 1 (medium cost), 2 (high cost) and 3 (very high cost).

<figure style='text-align:center'>
    <img src='img/Target.png' alt='data' style="width:100%"/>
    <figcaption style='text-align:center'>Figure 2: Target Feature</figcaption>
</figure>

In Figure 2 we can observe the Data Profile of our dataset. In this case we have a balanced target for training set, i.e., each class has almost the same representation. This is important because it makes it easier to create a general model using classical
metrics such as Accuracy or ROC-AUC.

### Access
To access data in our Workspace, we upload it from *train.csv* local file. We 
set the upload parameters (Tabular Data, separated by commas, with header, etc.).

This way, if we go to *Datasets* tab in ML Studio, we can see our Dataset and
watch a Profile as the one shown in Figure 2 and even get a Python SDK chunk to 
consume this dataset.

<figure style='text-align:center'>
    <img src='img/Dataset-upload.png' alt='data' style="width:100%"/>
    <figcaption style='text-align:center'>Figure 3: Datasets Section After Uploading Data</figcaption>
</figure>

<figure style='text-align:center'>
    <img src='img/Dataset-detail.png' alt='datadetail' style="width:70%"/>
    <figcaption style='text-align:center'>Figure 4: Details and Available Options for Uploaded Dataset.</figcaption>
</figure>

## Automated ML
The first model obtained is the one using AutoML. To do it, we use the automl.ipynb
file. In this notebook, we use the Python SDK to load the registered dataset and
configure an AutoML run. After that, the best model obtained is saved and if the 
results in terms of Accuracy are better than those obtained using Hyperparameter
Tuning, then the model is deployed and consumed as an API.

As part of the AutoML Configuration, it is used a Compute Target STANDARD_D2_V2 
with 4 `max_nodes`. This Compute Target is called `automl-mobiles`.

<figure style='text-align:center'>
    <img src='img/ComputeCluster.png' alt='compute' style="width:70%"/>
    <figcaption style='text-align:center'>Figure 6: Azure Compute Cluster for AutoML Run.</figcaption>
</figure>



### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
