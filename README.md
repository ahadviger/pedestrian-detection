# pedestrian-detection

Projekt iz predmeta Raspoznavanje uzoraka, ak. god. 2015./16.

* `python train.py train_model --descriptor [css|hog|hogcss] --model [ime datoteke] --hard_examples hard_example_1 hard_example_2`
Trenira SVM koristeći CSS/HOG i sprema serijalizirani model u data/[ime datoteke], koristeći slike iz data/window/train i data/window/test, gdje (u gornjem slučaju) uzima sve originalne slike i slike hard exampleova čije ime počinje s hard_example_1 ili hard_example_2.

* `python train.py prepare_initial --descriptor [css|hog|hogcss]`
Priprema inicijalne slike (centrirane pozitivne i 10x negativnih) i sprema u data/window/train i data/window/test.

* `python train.py prepare_hard_examples --descriptor [css|hog|hogcss] --model [ime datoteke] --prefix hard_example_3`
Priprema hard examples slike koristeći model u datoteci data/[ime datoteke] i sprema slike u data/window/train i data/window/test s prefixom hard_example_3 (u gornjem slučaju).

* `python detection.py --descriptor [css|hog|hogcss] --model [ime datoteke] --input [direktorij s ulaznim slikama] --output [direktorij s rezultatnim slikama]`
Prolazi kroz sve slike u [direktorij s ulaznim slikama], detektira pješake i koristeći model u datoteci data/[ime datoteke] i sprema po 4 rezultatne datoteke (npr. ime_slike.png):
  * ime_slike_true.png: točno označeni pješaci
  * ime_slike_all_detected.png: označeni pješaci našim programom prije non-maximum suppression
  * ime_slike_detected.png: označeni pješaci našim programom nakon non-maximum suppression
  * ime_slike_annotations.txt: tekstualna datoteka koja sadrži pozicije točno označenih pješaka i pozicije pješaka označenih našim programom (nakon non-maximum suppression)
