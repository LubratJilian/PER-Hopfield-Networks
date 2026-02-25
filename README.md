# PER - Hopfield Networks

## Présentation

https://github.com/user-attachments/assets/b7a426d1-d7a0-4532-8e9c-e293d144d4d6

## Structure du projet

Le dépôt contient :

* le **notebook principal** contenant l'ensemble du code utilisé pour ce projet ;
* le **poster** de présentation du projet de recherche ;
* le **rapport** du projet de recherche ;
* un **état de l’art** autour des réseaux de Hopfield ;
* un **package Python** correspondant à l’extraction du code principal du notebook.

## Fonctionnalités du notebook

Le notebook permet d’explorer :

* les réseaux de Hopfield classiques ;
* les méthodes de “rêve” associées ;
* les réseaux de Hopfield modernes.

Il contient notamment :

* préparation et transformer des jeux de données d’images (notamment MNIST) pour les rendre compatibles avec l’implémentation des réseaux de Hopfield ;
* entrainement et tester un réseau de Hopfield classique (mises à jour **synchrones** et **asynchrones**) ;
* génération des entrées bruitées et observer les capacités de reconstruction du réseau ;
* analyse certains comportements du réseau (**cycles**, **convergence**, **stabilité**) ;
* visualisation des trajectoires énergétiques et différents résultats expérimentaux (graphes, GIF, etc.) ;
* comparaison plusieurs configurations de réseaux sur différents jeux de données :
  * **MNIST**
  * **ImageNet 16x16**
  * **dataset décorrélé**
* implémentation et évaluation d'une version classique et moderne du réseau de Hopfield avec différentes fonctions d’énergie ;
* exploration d'une variante des réseaux classiques avec mécanisme de **“rêve”** pour l’étude de l’optimisation de la mémoire.

## Installation

### 1) Cloner le dépôt

```bash
git clone https://github.com/LubratJilian/PER-Hopfield-Networks.git
cd PER-Hopfield-Networks
```

### 2) Installer les dépendances

Installez les dépendances Python avec le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

> Conseil : vous pouvez utiliser un environnement virtuel (`venv`) avant l’installation pour éviter les conflits de versions.
>
> Exemple (optionnel) :
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Linux / macOS
> .venv\Scripts\activate    # Windows
> pip install -r requirements.txt
> ```

## Exécution

Le projet se lance via un **notebook Jupyter**.

### Lancer Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

Ensuite :

1. ouvrez le notebook du projet ;
2. exécutez les cellules dans l’ordre ;
3. suivez le flux d’expériences (prétraitement, apprentissage, bruit, reconstruction, visualisations, comparaisons).



## Utilisation du package Python (`hopfield_lib`)

Comme indiqué dans la partie *Structure du projet*, le dépôt contient une version **modularisée** du code sous forme de package Python : `hopfield_lib`.  
L’objectif est de pouvoir réutiliser l’implémentation des réseaux de Hopfield (classique, moderne, dreaming) sans dépendre du notebook.

### Structure (package + exemples)

Le package se trouve dans :

* `hopfield_lib/` : code réutilisable (réseaux + utilitaires)
* `hopfield_lib/tests/` : scripts d’exemples reproductibles

### Lancer un exemple

Important : lancez les scripts **depuis la racine du dépôt** (`PER-Hopfield-Networks/`).

#### Exemple 1 — Hopfield classique

Ce script entraîne un réseau de Hopfield sur quelques patterns binaires, bruite une entrée, puis reconstruit et affiche les résultats.

```bash
python -m hopfield_lib.tests.hopfield_classic
```

#### Exemple 2 — Trajectoire de convergence (énergie + GIF)

Ce script reconstruit une entrée bruitée en conservant la trajectoire des états, calcule l’énergie à chaque itération, détecte un éventuel cycle, et génère un GIF `convergence.gif`.

```bash
python -m hopfield_lib.tests.trajectory
```

#### Exemple 3 — Hopfield moderne

Ce script utilise la version moderne, avec une fonction d’énergie `f` configurable (par exemple : `x^10`).

```bash
python -m hopfield_lib.tests.hopfield_modern
```

#### Exemple 4 — Dreaming / matrice projecteur

Ce script applique une phase de “rêve” (matrice projecteur) après l’apprentissage, puis teste la reconstruction.

```bash
python -m hopfield_lib.tests.dreaming
```

### Utiliser la librairie

Une fois les dépendances installées, vous pouvez importer directement les classes et fonctions dans vos scripts Python :

```python
from hopfield_lib.NOM_FICHIER import CLASSE_OU_FONCTION
```

Exemples :

```python
from hopfield_lib.hopfield_network import HopfieldNetwork
from hopfield_lib.hopfield_network_modern import HopfieldNetworkModern
from hopfield_lib.hopfield_network_dream import HopfieldNetworkDream
from hopfield_lib.utils import bruitage_pixels_fixe
```
