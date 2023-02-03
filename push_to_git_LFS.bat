cd "C:\Users\biswa\OneDrive\Documents\Python Tutorial\Streamlite\NeuroHack"


call git lfs track "*.sav" "*.pkl" "*.csv" "*.xls" "*.pickle" "*.hdf5" && git add .gitattributes && git init && git add . && git commit -m "update" && git remote set-url origin https://github.com/SanjuktaBiswal/oddlogic.git && git push origin master --force

