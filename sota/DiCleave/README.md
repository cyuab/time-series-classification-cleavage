# Test
```
# Activate the corresponding environment
conda activate Dicleave

cd sota/DiCleave/DiCleave-main

# Train
python dicleave_t.py --mode 3 --input_file ../training_3p.csv --data_index 34657 --output_file ../models

# Predict
python dicleave.py --mode 3 --input_file ../test_3p.csv --data_index 3465 --output_file ../model_1_result.txt --model_path ../models/model_1.pt
# Apply the same for model_2.pt and model_3.pt
```