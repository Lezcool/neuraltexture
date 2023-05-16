#!/bin/bash
source /home/lez/Documents/PhD/generative/neuraltexture/lowtorch/bin/activate

cd /home/lez/Documents/PhD/generative/neuraltexture/code


path0='/home/lez/Documents/PhD/generative/neuraltexture/trained_models/'  #pretrained
path1='/home/lez/Documents/PhD/generative/neuraltexture/my_trained_models'
en0='test'
en1='neural_texture' 
# python ./test_neural_texture.py --trained_model_path "$path1" --experiment_name "$en1"
# python ./test_neural_texture.py --once --trained_model_path "/home/lez/Documents/PhD/generative/neuraltexture/trained_models/neural_texture/version_469408_neuraltexture_wood_3d_single"
# python ./train_neural_texture.py
python ./test_neural_texture.py --trained_model_path "$path1" --experiment_name "$en1"