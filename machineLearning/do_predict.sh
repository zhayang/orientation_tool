
datadir="/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/"
infile=$datadir"BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv"
outfile=$datadir"test_prediction_colinA.csv"
cfgfile=$datadir"test_stickslip_proceesing.json"

/home/zhay/anaconda3/envs/geophy/bin/python classify_stickslip.py -inFile $infile -outFile $outfile -cfgFile $cfgfile
