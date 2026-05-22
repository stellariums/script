for f in STRU_*; do
    num=${f#STRU_}
    d=SCF_${num}
    mkdir -p "$d"
    cp "$f" "$d/STRU"
done

for d in SCF_*; do
    cp ../INPUT ../KPT "$d"/
done