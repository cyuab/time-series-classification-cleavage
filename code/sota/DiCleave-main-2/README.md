# Test
```
# Activate the corresponding environment
conda activate Dicleave

# Change the working directory
cd sota/DiCleave/DiCleave-main

## 5p
# Train
python dicleave_t.py --mode 5 --input_file ../training_5p.csv --data_index 34657 --output_file ../models_5p

# Predict
python dicleave.py --mode 5 --input_file ../test_5p.csv --data_index 3465 --output_file ../models_5p_1_result.txt --model_path ../models_5p/model_1.pt
python dicleave.py --mode 5 --input_file ../test_5p.csv --data_index 3465 --output_file ../models_5p_2_result.txt --model_path ../models_5p/model_2.pt
python dicleave.py --mode 5 --input_file ../test_5p.csv --data_index 3465 --output_file ../models_5p_3_result.txt --model_path ../models_5p/model_3.pt


## 3p
# Train
python dicleave_t.py --mode 3 --input_file ../training_3p.csv --data_index 34657 --output_file ../models_3p

# Predict
python dicleave.py --mode 3 --input_file ../test_3p.csv --data_index 3465 --output_file ../models_3p_1_result.txt --model_path ../models_3p/model_1.pt
python dicleave.py --mode 3 --input_file ../test_3p.csv --data_index 3465 --output_file ../models_3p_2_result.txt --model_path ../models_3p/model_2.pt
python dicleave.py --mode 3 --input_file ../test_3p.csv --data_index 3465 --output_file ../models_3p_3_result.txt --model_path ../models_3p/model_3.pt

## multi
# Train
```