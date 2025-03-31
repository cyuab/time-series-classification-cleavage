# Test
```
# Activate the corresponding environment
conda activate Dicleave

cd sota/DiCleave/DiCleave-main

# Train
python dicleave_t.py --mode 3 --input_file ./dataset/training/training_3p.csv --data_index 34657 --output_file ../models

# Predict
python dicleave.py --mode 3 --input_file ./dataset/test/test_3p.csv --data_index 3465 --output_file ../result.txt --model_path ../models/model_1.pt
```