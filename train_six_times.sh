#!/bin/bash
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 1'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 1
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 2'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 2
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 3'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 3
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 4'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 4
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 5'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 5
echo 'python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 6'
python accent_cls_aesrc.py --gpu_id 0 --config configs/aesrc_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 6

echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 1'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 1
echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 2'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 2
echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 3'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 3
echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 4'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 4
echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 5'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 5
echo 'python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 6'
python accent_cls_vctk.py --gpu_id 0 --config configs/vctk_wavlm_cont_disc_mhubert_cont_disc.yaml --seed 6
