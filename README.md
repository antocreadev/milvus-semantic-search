# Similarité de mots avec Milvus et Spacy

Ce projet implémente une recherche de similarité de mots basée sur des vecteurs d'embeddings à l'aide de [Milvus](https://milvus.io/) pour l'indexation vectorielle et de [Spacy](https://spacy.io/) pour la génération des embeddings.

Ce travail a été réalisé dans le cadre du cours "Base de données non relationnelles" (M1DIRA5-BD-NON-RELATIONNELLE) dispensé par Évelyne Vittori à l'Université de Corse Pasquale Paoli.

L'objectif de l'exercice était de former des binômes au sein de la promotion, afin d'explorer, tester et documenter un système de gestion de bases de données (SGBD) NoSQL en deux heures. Les résultats obtenus devaient ensuite être présentés devant l'ensemble de la promotion.

## Fonctionnalités

- Création d'une collection vectorielle dans Milvus.
- Indexation de mots et de leurs embeddings vectoriels.
- Recherche des mots les plus similaires à une entrée utilisateur en fonction de leurs embeddings vectoriels.
- Interactive : l'utilisateur peut entrer des mots et obtenir leurs correspondances les plus proches.

## Prérequis

- Python 3.8 ou supérieur.

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/antocreadev/milvus-semantic-search.git
```
2. Se déplacer dans le répertoire du projet :
```bash
cd milvus-semantic-search
```

3. Créez un environnement virtuel :
```bash
python3 -m venv .venv
```

4. Activez l'environnement virtuel :
- Pour Linux & MacOS :
```bash
source .venv/bin/activate
```
- Pour windows :
```bash
.venv\Scripts\activate
```

5. Mettre à jour pip
```bash
pip install --upgrade pip
```

6. Installez les dépendances :
```bash
pip install -r requirements.txt
```

7. Téléchargez le modèle de langue Spacy :
```bash
python -m spacy download fr_core_news_md
```

8. Lancer Milvus : 
```bash
make run
```


## Utilisation
- Exécutez le script principal :
```bash
python3 main.py
```

- Supprimer l'image Docker de Milvus :
```bash
make delete
```


