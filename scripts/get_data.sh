wget -c https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train
wget -c https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa
wget -c https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb

mkdir -p conll2003
mv eng.* conll2003
