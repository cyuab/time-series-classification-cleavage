# Related SOTA (state-of-the-art) models 
(With the latest commit, accessed on 2025-06-17)
- [DiCleavePlus](https://github.com/MGuard0303/DiCleavePlus) (42f4853)
- [DiCleave](https://github.com/MGuard0303/DiCleave) (e512d74)
- [ReCGBM](https://github.com/ryuu90/ReCGBM) (018f7a7)

We have downloaded their latest version (commit) and placed in this folder.

```
conda create -n dicleave python=3.11.3 numpy pandas scikit-learn pytorch -c conda-forge -c pytorch
conda activate dicleave
ls
# code            data            figures         README.md       results <--- In the folder directory
cd code/sota
conda env export > dicleave.yml
unzip 'DiCleave-main-e512d74.zip'
cd DiCleave-main
# 5p
# Train, repeat for training_5p_i, where i = 0..4
python dicleave_t.py --mode 5 --input_file ../DiCleave-data/training_5p_0.csv --data_index 34657 --output_file ../DiCleave-data/training_5p_0_model
# Test, repeat for training_5p_i, where i = 0..4
python dicleave.py --mode 5 --input_file ../DiCleave-data/test_5p_0.csv --data_index 3465 --output_file ../DiCleave-data/training_5p_0_model/model_1_result.txt --model_path ../DiCleave-data/training_5p_0_model/model_1.pt

# Apply the same for 3p and multi
# Change "-mode 5" to "-mode 3" and "-mode multi" correspondingly 
```