clear
rm main.out ./classification/labelsVIM3Paralelo.csv
g++ -Wall main.cpp functions.cpp -o  main.out -lOpenCL -std=c++11
time ./main.out
echo ""
echo "=====     =====     =====     ====="
echo "Comparison betweenSerial vs. openCL codes:"
python3 file_comparison.py
echo "=====     =====     =====     ====="
rm main.out
