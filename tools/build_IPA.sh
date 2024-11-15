
CUDA_VISIBLE_DEVICES=0 python -u build_corda.py --model_id="meta-llama/Llama-2-7b-hf" \
    --cov_aware --r $1 \
    --use_cache --save_model --save_path $2 \
    --first_eigen --calib_dataset "MetaMATH"

# Cov on triviaqa: --calib_dataset "traivia_qa" \
# Cov on nqopen:   --calib_dataset "nqopen" \
# Cov on MetaMATH: --first_eigen --calib_dataset "MetaMATH" \  
# Con on codefb:   --first_eigen --calib_dataset "codefeedback" \
# Cov on Wizard:   --first_eigen --calib_dataset "WizLMinstruct" \
# Cov on Alpaca:   --first_eigen --calib_dataset "alpaca" \
