BASEDIR=$(dirname "$0")
find "$BASEDIR" -type f -name "*.ipynb" -print -exec jupyter nbconvert --to notebook --execute --inplace {} \;