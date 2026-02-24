# PER - Hopfield Networks

## Présentation

Ce projet est une implémentation (sous forme de notebook) de plusieurs variantes de **réseaux de Hopfield** pour des tâches de mémorisation et de reconstruction de motifs, dans le cadre du PER **2025-023** de l'école Polytech Nice Sophia, intitulé:

**Rêver pour mieux mémoriser : optimisation mémoire dans les réseaux de Hopfield**

Le notebook permet d’explorer :

* les réseaux de Hopfield classiques
* les méthodes de “rêve” associées
* les réseaux de Hopfield modernes

## Fonctionnalités

Le projet permet notamment de :

* préparer et transformer des jeux de données d’images (notamment MNIST) pour les rendre compatibles avec l'implémentation des réseaux de Hopfield 
* entraîner et tester un réseau de Hopfield classique (mises à jour **synchrones** et **asynchrones**) 
* générer des entrées bruitées et observer les capacités de reconstruction du réseau
* analyser certains comportements du réseau (**cycles**, **convergence**, **stabilité**)
* visualiser des trajectoires énergétiques et différents résultats expérimentaux (graphes, GIF, etc.)
* comparer plusieurs configurations de réseaux sur différents jeux de données :

  * **MNIST**
  * **ImageNet 16x16**
  * **dataset décorrélé**
* implémenter et évaluer une version moderne du réseau de Hopfield avec différentes fonctions d’énergie
* explorer une variante avec mécanisme de **“rêve”** pour l’étude de l’optimisation de la mémoire

---

## Structure du projet

Le dépôt contient :

* le **notebook principal** (code de l’implémentation et expériences)
* le **rapport** du projet
* un **état de l’art** autour des réseaux de Hopfield

## Installation

### 1) Cloner le dépôt

``` bash
git clone [https://github.com/LubratJilian/PER-Hopfield-Networks.git](https://github.com/LubratJilian/PER-Hopfield-Networks.git)
cd PER-Hopfield-Networks
```

### 2) Installer les dépendances

Installe les dépendances Python avec le fichier `requirements.txt` :

``` bash
pip install -r requirements.txt
```

> Conseil : tu peux utiliser un environnement virtuel (`venv`) avant l’installation pour éviter les conflits de versions.
>
>Exemple (optionnel) :
>```bash
>python -m venv .venv
>source .venv/bin/activate   # Linux / macOS
>.venv\Scripts\activate    # Windows
>pip install -r requirements.txt
>```

## Exécution

Le projet se lance via un **notebook Jupyter**.

### Lancer Jupyter

```
jupyter notebook
```

ou

```
jupyter lab
```

Ensuite :

1. ouvrir le notebook du projet ;
2. exécuter les cellules dans l’ordre ;
3. suivre le flux d’expériences (prétraitement, apprentissage, bruit, reconstruction, visualisations, comparaisons).
