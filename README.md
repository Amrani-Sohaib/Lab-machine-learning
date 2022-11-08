
# Lab-machine-learning
Repo pour lab machine learning

Ce projet à pour but d'entrainer un modèle de régression logistique il se compose de 3 parties principales.

partie 1: Importation des données et preprocessing
Import des données et structuration des matrices X et y puis on applique du text préprocessing pour avoir un text clean est avoir une vectorisation qui prend en compte des mots qui portes de la valeur sémantique et rendre le modèle plus performant.

Partie 2: Entrainement du modèle
On initialise un vectorizer du type Tf-Ifd et un classifieur de type régression logistique avec scikit-learn puis on les stocks dans une pipeline car c'est plus pratique, on initialise aussi un dictionnaire de hyperparamètres à optimiser.
Après entrainement du modèle on check les résultats avec le score d'accuracy et la matrice de confusion puis le rapport de classification.

Pour répondre à la question de comment améliorer le modèle, on peut s'y prendre avec plusieurs approches
- faire une optimisation des hypèreparamètres avec une Gridsearch, Randomsearch ou Bayesiansearch (on l'a run en fin de projet)
- tester d'autres classifieurs (on a testé plusieurs classifieurs et on a décidé de garder le Ridge le PassiveAggressive) 
- faire appel à des techniques d'ensembles notamment le stacking pour utiliser plusieurs weak classifiers (pour la stack choisi c'est un 3rl + 3ridge + PassiveAggressive avec une régression logistique comme éstimateur finale)
On constat l'évolution des performances (On a pas fait d'optimisation d'hyperparamètres par manque de ressources de calculs)

Partie 3: Extraction des 10 mots les plus lourd dans la règle de décision

