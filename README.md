# Team Members
* Harikrishnan Chalapathy Anirudh
* Swastik Majumdar
* Nguyen Thai Huy
* Hayden Ang Wei En 
* Win Tun Kyaw

# Introduction
Online hate speech is an important issue that breaks the cohesiveness of online social communities and even raises public safety concerns in our societies. Motivated by this rising issue, researchers have developed many traditional machine learning and deep learning methods to detect hate speech on online social platforms automatically.

Essentially, the detection of online hate speech can be formulated as a text classification task: "Given a social media post, classify if the post is explicitly hateful, implicitly hateful or non-hateful". In this project, we apply 6 different machine learning models to perform hate speech classification.

# Highlights of the models used
* **Extra Trees**

Our **_best_** performing model. The core idea behind Extra-Trees Classifier is to fit n number of randomized decision trees with several sub-samples of our dataset, which then employs averaging to improve the accuracy of our model while controlling over-fitting of the training dataset. Although a single decision tree might be a weak classifier, we introduce variations into the decision tree by randomly selecting a random subset of features when building each node of a tree. By putting n number of randomized decision trees together, a single low accuracy classifier is then able to be turned into a high performing forest.

*	**Logistic Regression**

Logistic Regression predicts the probability of occurrence of a binary event by utilizing a logit function. Although, this model is traditionally a model used to classify binary classes, the sklearn library offers the option for multiclass classification. However, since our problem statement requires to perform binary classification, we do not need this multiclass option.

*	**Gradient Boosting**

The Gradient Boosting model is an additive model, allowing for optimization of an arbitrary differentiable loss function. It works by combining weak learning models together to create a strong predictive model, most often, using decision trees similar to Extra-Trees Classifier and Random Forest Classifier. What separates them from one another is that in Gradient Boosting classifiers, the trees are trained sequentially, where each tree is trained to correct the errors of the previous ones and when determining the output of each tree, Gradient Boosting classifiers have to check each tree in a fixed order.

*	Voting Classifier

A voting classifier is a ML model that trains an ensemble of numerous models, then predicts an output by aggregating the findings of each classifier passed into the Voting Classifier. It predicts the output class based on the highest majority of voting. It supports two types of voting, hard and soft, where hard voting is just the aggregation of the output of each class and selecting the class with the majority of votes. Whereas for soft voting, the summation of probabilities for each class from all models is taken into consideration, and the output class is determined by the class that had the highest probability.

*	**Bagging Classifier**

A bagging classifier is an ensemble meta-estimator that fits base classifiers each with a random subset of the original dataset then aggregating the individual predictions similar to the voting classifier to arrive at the final output.

*	**Random Forest Classifier**

Random Forest Classifier is very similar to the Extra-Trees Classifier, both composed of many decision trees, where the final output is determined by considering the prediction of all trees. However, where they differ is in the selection of subsample of the dataset when partitioning each node. Random Forest uses bootstrap replicas where it subsamples input data with replacement, this causes increase in variance as bootstrapping makes it more diversified. Another difference is the selection of cut points in order to split the nodes of the trees. Random Forest chooses the optimum split while Extra-Trees chooses it randomly.

# Post-Mortem
We noticed that the training dataset was imbalanced, with significantly more examples of class ‘0’ compared to class ‘1’. Additionally, many initial models predicted a lot of the provided test features to be class ‘0’; clearly, the skewed dataset was causing them to perform poorly. Hence, we researched possible solutions and decided to use a data pre-processing method from imblearn called the Synthetic Minority Oversampling Technique (SMOTE).

This technique synthetically generates new data entries in the training set, from the minority class (in this case, class ‘1’), that are slightly different from existing entries until the two classes are at a 1:1 ratio. The advantages of using SMOTE are that firstly, no information is lost during data pre-processing. This is because all original data entries in the training set are kept. Secondly, variance in the dataset is preserved as the technique adds entries that are only similar to the original, and not duplicates. Finally, overfitting due to imbalanced dataset is reduced.

Such data pre-processing techniques should be taught in future Machine Learning courses. This is because datasets are likely to be imbalanced in real life. Knowing the proper data pre-processing techniques, and when to use the appropriate ones, would enhance one’s effectiveness as a data scientist and improve the models that they create. This would also improve students’ understanding of not only the limitations of using machine learning models and how to overcome them, but also the concepts of underfitting and overfitting. Additionally, introduction to concepts such as hyperparameter tuning will surely help Machine Learning engineers in the future.
