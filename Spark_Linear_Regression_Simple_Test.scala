// Databricks notebook source
// MAGIC %md
// MAGIC # Test de l'algorithme de régression linéaire de Spark MLlib
// MAGIC 
// MAGIC *L'algorithme de régression linéaire fait partie des algorithmes classiques de l'apprentissage automatique ou machine learning. L'objectif est de trouver la droite, dans le plan, ou dans un espace multi-dimensionnel, qui pourrait porter l'ensemble des points d'un jeu de données.*
// MAGIC 
// MAGIC Ce notebook a pour objectif d'illustrer:
// MAGIC - la démarche de la détermination d'un modèle à partir d'un jeu de données une fois un algorithme choisi
// MAGIC - l'utilisation des API Spark MLlib, via le langage de programmation Scala
// MAGIC - l'utilisation de la solution Saas de Databricks pour la réalisation de Notebooks sur Spark : https://community.cloud.databricks.com
// MAGIC 
// MAGIC 
// MAGIC Nous allons utiliser un jeu de données créé spécifiquement pour ce test, en plaçant tous les points sur une même droite:
// MAGIC 
// MAGIC `
// MAGIC y = f(x), où f(x) = x pour tout réel x.
// MAGIC `
// MAGIC 
// MAGIC Les étapes:
// MAGIC - Création d'un jeu de données basé sur une fonction mathématique connue.
// MAGIC - Utilisation du jeu de données pour entrainer un algorithme de régression linéaire et observer le resultat trouvé.
// MAGIC - Utilisation du modèle pour une prédiction à partir d'une valeur pour laquelle on connait le résultat

// COMMAND ----------

// MAGIC %md
// MAGIC **Auteur: Marc Durocher**
// MAGIC 
// MAGIC **Date de création: 18 Septembre 2016**
// MAGIC 
// MAGIC **Date de mise à jour: 09 Octobre 2016**

// COMMAND ----------

import org.apache.spark.sql.SQLContext
import scala.math._

/*
 * Le jeu de données est créé à partir de intervalle d'entiers naturels [1; 10 000]
 * L'ensemble des entiers de 1 à 10 000 est transformé en ensemble de "Double" (rééls) d'un point de vue informatique (1er map)
 * Puis on créé finalement un ensemble de couple (x,x) pour to x de l'ensemble de rééls précédemment produit (2ème map)
 */

val df = sqlContext.createDataFrame(1 to 10000 map {_.toDouble} map { r => (r+50*sin(r/100f),r)} ).toDF("label","x")

/*
 * Affichage du schema de la Dataframe (c'est une vue sous forme de table)
 */
df.printSchema()

// COMMAND ----------

/*
 * Affichage des premiers éléments de la dataframe (ou table)
 */

df.show(11)

// COMMAND ----------

/*
 * La dataframe est concrétement transformée en table afin de pouvoir exécuter des requêtes SQL
 * La table créée est nommée `maTable`
 */
df.registerTempTable("maTable")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Aperçu des données
// MAGIC 
// MAGIC Les données présentes en table sont requêtées en SQL:
// MAGIC 
// MAGIC `select * from maTable`
// MAGIC 
// MAGIC Puis on utilise la capacité du notebook à représenter graphiquement les données. Choix du mode de représentation "Scatter Plot", dans "Plot Options...".
// MAGIC 
// MAGIC *Vérification que l'on a bien la fonction affine x -> x*

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select * from maTable

// COMMAND ----------

// MAGIC %md
// MAGIC ## Recherche d'un modèle avec l'algorithme de régression linéaire.
// MAGIC 
// MAGIC On utilise l'algorithme implémenté dans Spark MLlib (Machine Learning library).
// MAGIC 
// MAGIC La première étape consiste à rassembler toutes les dimensions (les différents axes, ou paramètres de la fonction) dans un vecteur, car c'est un vecteur qui est attendu, pour chaque point, en entrée de l'algorithme.
// MAGIC Ici nous n'avons qu'une seule dimension (l'axe des "x"). Nous transformons donc chaque élément x en vecteur à une dimension (x).
// MAGIC 
// MAGIC On utilise l'objet VectorAssembler qui assemble différentes "dimensions" (ici des "colonnes") en un unique vecteur multi-dimensionnel (une dimension pour ce cas).
// MAGIC La colonne résultat est nommée `features` car c'est le nom attendu par l'algorithme de régression linéaire.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
val vecAssembler = new VectorAssembler()

val features = vecAssembler
  .setInputCols(Array("x"))
  .setOutputCol("features")  // On attribue le nom `features` à la colonne qui contient les vecteurs créés 
  .transform(df)


/*
 * Affichage de la dataframe résultante
 */
features.show(11)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Entrainement du modèle
// MAGIC 
// MAGIC Une instance d'objet LinearRegression est créée.
// MAGIC 
// MAGIC On indique un nombre d'itération maximum (c'est une démarche itérative pour trouver la fonction linéaires qui "colle" le plus aux données): ici on choisit 10 itérations.
// MAGIC 
// MAGIC Le paramètre de régularisation est placé à 0,0001. https://fr.wikipedia.org/wiki/R%C3%A9gularisation_(math%C3%A9matiques)
// MAGIC 
// MAGIC Puis on entraine le modèle avec le jeu de données (`features`).
// MAGIC 
// MAGIC Finalement on écrit les coefficients trouvés, suite à cet entraînement (un coefficient par dimension, donc ici, un seul coefficient) et la valeur de l'ordonnée à l'origine.
// MAGIC 
// MAGIC ` f(x) = a . x + b` 
// MAGIC 
// MAGIC a: le coefficient (... de la pente de la droite)
// MAGIC 
// MAGIC b: l'ordonnée à l'origine (*intercept* en anglais)

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.0001)

// Entraine le modèle
val lrModel = lr.fit(features)

// Affiche le(s) coefficient(s) et la valeur à l'ordonnée
println(s"Coefficients: ${lrModel.coefficients(0)} \n Valeur à l'ordonnée: ${lrModel.intercept}\n")


// COMMAND ----------

// MAGIC %md
// MAGIC ## Test de prédiction
// MAGIC 
// MAGIC Maintenant que notre modèle est établi, nous pouvons utiliser les coefficients spécifiques à notre modèle pour réaliser des prédictions.
// MAGIC Pour une valeur inconnue x, la valeur attendue est `a . x + b`
// MAGIC où
// MAGIC 
// MAGIC `a= lrModel.coefficients(0)`
// MAGIC 
// MAGIC et
// MAGIC 
// MAGIC `b= lrModel.intercept`
// MAGIC 
// MAGIC On crée un fonction `prediction` dédiée au calcul prédictif:
// MAGIC 
// MAGIC `
// MAGIC def prediction(x:Double) =  lrModel.coefficients(0) * x + lrModel.intercept
// MAGIC `
// MAGIC 
// MAGIC (\*) *la pente trouvée est 0,99999 et la valeur à l'ordonnée est 0,0002*
// MAGIC 
// MAGIC 
// MAGIC Et pour x=678 quelle est la valeur prédite ? (on s'attend à une valeur proche de 678)

// COMMAND ----------

import scala.math._

/*
 * Création d'une simple fonction Scala pour calculer a.x+b
 */
def prediction(x:Double) =  lrModel.coefficients(0) * x + lrModel.intercept

/*
 * Calcul de la valeur prédite par notre modèle pour la valeur x=678
 */
val resultatPredit = prediction(678.toDouble)

/*
 * Le résultat attendu est 678
 */
val resultatAttendu = (678).toDouble 

/*
 * Calcul de l'erreur entre la valeur attendue et la valeur prédite par notre modèle
 */
val erreur= abs(resultatAttendu -  resultatPredit)

println(s"La valeur prédite pour 678 est $resultatPredit, avec un écart de $erreur")


// COMMAND ----------

// MAGIC %md
// MAGIC Fin du cahier d'exercice...
// MAGIC 
// MAGIC Marc Durocher
