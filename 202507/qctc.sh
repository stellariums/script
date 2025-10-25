#!/bin/bash
OUTPUT_FILE="qctc.data"
> "$OUTPUT_FILE"
echo -e "Directory\tThermal Conductivity (k)\tSpectral Thermal Conductivity (k_spec)\tQuantum Corrected Thermal Conductivity (k_qc)" > "$OUTPUT_FILE"
for dir in */;
    do
    if [ ! -d "$dir" ]; then
        continue
    fi

    # ½øÈë×ÓÎÄ¼þ¼Ð
    echo "Processing directory: $dir"
    cd "$dir"

    # ¼ì²éÊÇ·ñ´æÔÚ±ØÒªµÄÎÄ¼þ
    if [ ! -f "kappa.out" ]; then
        echo "Error: kappa.out not found in $dir"
        cd ..
        continue
    fi

    # ÔËÐÐ Python ½Å±¾
    read -r thermal_kappa spectral_kappa quantum_kappa < <(python3 ../qctc.py)

    # ½«½á¹û×·¼Óµ½Êä³öÎÄ¼þ
    echo -e "${dir%/}\t$thermal_kappa\t$spectral_kappa\t$quantum_kappa" >> "$OUTPUT_FILE"
    # ·µ»ØÉÏ¼¶Ä¿Â¼
    cd ..
done

echo "Processing complete. Results are saved in $OUTPUT_FILE"
