# Citation prediction of articles in theoretical physics


## Introduction and overview

## Datasets

The theoretical physics articles were collected on [ArXiV](https://arxiv.org/) using the associated [API](https://arxiv.org/help/api/basics). The acessible data was the names of the authors, the date of creation (when the authors uploaded the article) and the unique id number that ArXiV gives to each article. This part of the harvesting was performed in `get_data.py` [here](src/data/get_data.py).

In order to get the total citation count but also the number of citations that an article get each year after it is created, it was necessary to use the [InspireHep API](https://github.com/inspirehep/rest-api-doc) (InspireHEP is a database that gathers articles in high energy physics). For more information, please refer to the script `get_citations.py` [here](src/data/get_citations.py).

The initial project was to study all the articles uploaded from year 1995 to 2015, but since each API limited the number of requests per second, a sleeping time was sometimes mandatory, which rendered the process very long.  Due to time constraints, only the articles between 01-01-2010 and 01-01-2015 were considered.
Therefore, the full dataset comprises 7397 valid `hep-th` articles.

As a first observation, it's interesting to look at the distribution and the statistics of the total number of citations per article, as it is shown below.

<p>
  <img src="reports/figures/citation_distribution_2010_2015.png" width=35% height=35% >
</p>

|Count|Mean|Min|Max|75%|
|-----|----|---|---|---|
|7397|34|0|2958|38|

One can see that the large majority of articles receive little to no citations. The 75th percentile is already at 38 citations and the mean is only 34. The citation distribution for this dataset is long-tailed and corresponds more or less to a power law, as it was concluded in other [citation studies](https://arxiv.org/pdf/physics/0407137.pdf). Thus, it is important to account of this imbalanced classification when selecting the training and testing set. More on that in the next section.


## Methodology

### Features and sampling
The features used in the first place were simply the number of citations during the three years after the first version of was published. For instance, if some was article was uploaded first in the year 2010 (no matter the month), the numbers collected were the citations in 2010 to which we added the citations in 2011, then the citations in 2012 and finally the citations in 2013.  The line of thought here is that if an article gains some momentum in the first three years, then it's likely that it will be at least cited a lot.

The label associated to the three features is the total number of citations from the uploading year to 2022, which was divided into three unequal categories. Indeed, here the goal is not to get the precise number of citations an article will get in the future, that's why a rough classification was privileged.
The first category, namely **A**, include the articles that have a total of citations below the mean (34). The second category **B** corresponds to articles with citations between 34 and 100. The third and final category **C** are the articles with more than 100 citations. These categories follow respectively a (75,20,5) rule approximatively. Hence, the full set must be split into training/test sets accordingly. A `StratifiedShuffleSplit` was used instead of the usual `train_test_split` because the latter through its random sampling can introduce a bias in each set, meaning that some category could be overrepresented. The dataset was split in the 80/20 fashion.
The [script](src/models/sampling_features.py) `sampling_features.py` processes what has been discussed in the last two paragraphs. The next image shows the distribution of the three categories in the two sets.

<p>
    <img src="reports/figures/train_test_stratification_2010_2015.png" width=35% height=35%>
</p>

### Models

The choice of classifiers is pretty standard. For the time being, only a logistic regression, a decision tree and a random forest were implemented. The table of the parameter space explored with a 5-fold cross validation and the optimal hyperparameters is shown below. The F1-score was used to discriminate (see next subsection).

|        |Logistic Regression|Decision Tree|Random Forest|
|--------|-------------------|-------------|-------------|
|Parameter space|'solver' : ['saga','lbfgs','newton-cg']|'max_depth' : [x+1 for x in range(32)], 'min_samples_split' : [2, 5, 10, 20, 50, 100, 200], 'min_samples_leaf' : [1, 4, 7, 10] |'criterion' : ['gini','entropy'], 'max_depth' : [1, 5, 10, 50], 'max_features' : ['log2','sqrt'], 'n_estimators' : [100, 150, 200, 250, 300]|
|CV Result|'solver' : 'saga', 'max_iter' : 2000'|'max_depth' : 9, 'min_samples_split' : 50, 'min_samples_leaf' : 7|'criterion' : 'entropy', 'max_depth' : 5, 'max_features' : 'log2', 'n_estimators' : 150|

Note that some interval of the maximum iteration was also put to tune, but it gave rise to a convergence error. A number of iteration of a 2000 was found to be enough and did not lead to divergence.

### Metrics

In the spirit of a quick and dirty analysis, the F1-score was chosen to determine how good a classifier is. In an imbalanced setting, it is highly preferable to use this metric instead of accuracy, since the latter, by definition, will be close to 1 if one class is much more represented. Although it seems F1 does a good job


## Results
### Logistic Regression

### Decision Tree

### Random Forest

## Future implementations
