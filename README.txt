-Le projet contient :
	#Un dossier dataSet qui contient des enregistrements audio (.wav) des mots "up" et "down" en "darija".
	#Un fichier python p1 sert a creer un fichier data.json qui contient les donnees (MFCCs) de dossier dataSet.
	#Un fichier python p3 qui sert a traiter nos donnees en utilisant l'apprentissage automatique (Machine learning).
 
-Pour tester notre projet vous pouvez utiliser :
	#la fonction "checkVoiceByPath(path)" en passant en parametre le lien vers un fichier .wav.
	#vous pouvez changer les parametres de la fonction train_test_split et voir le resultat final.
	#vous pouvez comparer notre vecteur de prediction avec le vecteur "labels" dans data.json qui contient les vrais valeurs.
 C'est mieux d'enregistrer ce fichier wav en utilisant un telephone portable pour avoir une bonne qualite.
 Ce fichier ne doit pas passer une seconde.

-Il faut lancer le script p1.py premierement pour generer le fichier data.json apres vous pouvez lancer le script p3.py
-Notre projet se base sur des packages de :
	#Traitement des fichier et des audios(os/librosa/numpy.json).
	#Machine learning(Tensorflow/sklearn).
	#Representation graphique(matplotlib).
 Si vous n'avez pas ces packages sur votre ordinateur vous pouvez l'installer par la command : pip install (nom du package)
 sinon vous pouvez utiliser le service de google COLAB qui peut vous offrir un ordinateur virtuel qui contient ces packages,
 c'est mieux de travailler dans colab avec un processeur graphique (GPU) pour gagner du temps.
 
	