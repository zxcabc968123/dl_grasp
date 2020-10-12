# Data_amplification
1. -> use labelme to get label.png 
2. -> put your "background img" to background/
3. -> put your "want to train img" to data/
4. -> files in data/ should have class you want to amplification and make sure each class folder contain img/ and label/
      make sure img.jpg and label.png have same name i.e. img/1.png label/1.png
5. -> python3 all.py (also you can use python3 all.py --num_each 2,0 --num 1)
6. -> the file you amplificate would exit in output
7. -> enjoy!!

total number of photo generated is num * number_of_background * length_of_num_each
