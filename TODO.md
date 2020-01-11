# TODO list

## Code

### Train / Validation

Séparer en train et validation, parce que la c'est train=train et validation=validation

### Reconstruction du spectrogramme à partir d'un batch de sortie

Chaque son est considéré comme un batch.

As it is, the model input and output have the same shape.
There are 4 ways to proceed then :

#1 Keep it as it is, but putting a stride (=step) params in
the batchify function, putting it equals to nframes,
s.t. there is no overlapping

#2 Keep it as it is, adding the stride params to the
batchify function, but with stride < nframes, thus making
overlapping between windows, which has to be taen into
account for the reconstruction

#3 Only tak one frame (the middle one) form the ouput, may
be the easiest, but then : do we compute the loss only on
this frame or on the whole output, even if only taking the
middle frame for reconstruction

#4 Finally, we may want to compute the loss on y, not Y,
such that whatever the reconstruction method used, it is
taken into account, because then the general loss on the
reconstructed spectro is what matters. It may be the best.

Il semble le plus versatile de faire l'option #4. Sachant que alors la
méthode de reconstruction fera partie du graph des computations,
autrement dit le gradient sera calculé dessus (si application de fenêtre
d'apodisation par exemple)

Il faut que cette fonction `reconstruct` marche de concert avec la
fonction `batchify`. Avec des arguments comme par exemple la position
de l'alignement (gauche droite ou milieu) qui va influer sur le padding
et un argument `stride` pour le nombre de frames sautées entre deux
windows.

### Reconstruction du signal à partir du sectrogramme

Faire une fonction de reconstuction du signal à partir du spectrogramme.
Pour cela, il faut avoir accès à l'angle du spectrogramme input pour le
réinjecter dans la fonction istft avec le module débruité.
Donc, il faut faire un batch_loader spécial pour l'étape de test,
qui permet d'avoir cette information là. L'angle est actuellement non
sorti par batch_loader, mais bien calculé, donc juste une petite
modification devrait suffire. Autrement, on peut le sortir dans tous les
cas, et ne pas le garder pour le train, ce qui peut êre le plus propre.

On peut aussi penser à entraîner un réseau en parallèle pour le
débruitage de la phase, ce qui ne devrait poser aucun problème puisque
tout serait alors exactement pareil.


## Training

Essayer un training avec un RSB décroissant

## Questions générales

Faut-t-il séparer ET le noise ET le raw e train val test ? I mean, peut
être que le faire pour le bruit suffit, c'est bien sa distribution à lui
qu'on veut apprendre, mais bon on l'apprend aussi au regard de la
distribution du raw donc...
Mais pour l'instant on sépare le raw en train et test, et le noise en
train val test. Surtout que en fait le babble noise à l'air d'avoir
une ditribution assez constante dans le temps.

## Mesures

RSB sortie vs RSB entrée --> permet de comparer différents réseaux sur
la même tâche.

Pour ça on peut faire :

RSB entre clean et bruit en sortie
Rapport en bruit en sortie et bruit enlevé