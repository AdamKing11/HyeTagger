wget https://dumps.wikimedia.org/hywiki/20161001/hywiki-20161001-pages-articles.xml.bz2

python2 WikiExtractor.py -o OUTPUT hywiki-20161001-pages-articles.xml

cat OUTPUT/A*/wiki_* > all.txt

cat all.txt  | grep -v "<doc" | grep -v "</doc" | grep -v "^$" | grep -v "Medi" > all.txt.clean
