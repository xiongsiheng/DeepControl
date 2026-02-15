
save_path=~/debug_cache

python download.py --savepath $save_path

cat $save_path/part_* > e5_Flat.index
