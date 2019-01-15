# databricks_notebook_ml_linear_1

## Test de l'algorithme de régression linéaire de Spark MLlib
L'algorithme de régression linéaire fait partie des algorithmes classiques de l'apprentissage automatique ou machine learning.
L'objectif est de trouver la droite, dans le plan, ou dans un espace multi-dimensionnel, 
qui pourrait porter l'ensemble des points d'un jeu de données.

Ce notebook a pour objectif d'illustrer:
- la démarche de la détermination d'un modèle à partir d'un jeu de données une fois un algorithme choisi
- l'utilisation des API Spark MLlib, via le langage de programmation Scala
- l'utilisation de la solution Saas de Databricks pour la réalisation de Notebooks sur Spark : https://community.cloud.databricks.com

*(Il faut importer le source .scala dans votre espace de travail sous Databricks. C'est une page de notebook)*

Nous allons utiliser un jeu de données créé spécifiquement pour ce test, en plaçant tous les points sur une même droite:

>
>
> y = f(x), où f(x) = x pour tout réel x.
>
>
Les étapes:
- Création d'un jeu de données basé sur une fonction mathématique connue.
- Utilisation du jeu de données pour entrainer un algorithme de régression linéaire et observer le resultat trouvé.
- Utilisation du modèle pour une prédiction à partir d'une valeur pour laquelle on connait le résultat
