for npseed in 1 2 3 4 5 6 7 8
do
for tfseed in 1 2 3 4 5
do 
echo "python loyo_testing.py mlp ${npseed} ${tfseed} > LOYO_results/seeds/log_loyo_mlp_2048_maeloss_y2019_np_${npseed}_tf_${tfseed}.log"
python loyo_testing.py mlp ${npseed} ${tfseed} > LOYO_results/seeds/log_loyo_mlp_2048_maeloss_y2019_np_${npseed}_tf_${tfseed}.log
done
done