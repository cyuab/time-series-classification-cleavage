# Test
```
# Activate the corresponding environment
conda activate Dicleave

cd sota/DiCleave/DiCleave-main

## 3p
# Train
python dicleave_t.py --mode 3 --input_file ../training_3p.csv --data_index 34657 --output_file ../models_3p

# Predict
python dicleave.py --mode 3 --input_file ../test_3p.csv --data_index 3465 --output_file ../3p_model_1_result.txt --model_path ../models_3p/model_1.pt
# Apply the same for model_2.pt and model_3.pt

## 5p
# Train
python dicleave_t.py --mode 5 --input_file ../training_5p.csv --data_index 34657 --output_file ../models_5p

# Predict
python dicleave.py --mode 5 --input_file ../test_5p.csv --data_index 3465 --output_file ../5p_model_1_result.txt --model_path ../models_5p/model_1.pt
# Apply the same for model_2.pt and model_3.pt
```