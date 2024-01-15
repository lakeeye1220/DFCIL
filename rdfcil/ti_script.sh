mkdir data # create data folder (if not delete it)
mkdir data/tiny-imagenet200 # create tiny-imagenet200 folder (if not delete it)
mkdir data/tiny-imagenet200/val
# create symbolic links
ln -s /data/tiny-imagenet/train/ data/tiny-imagenet200/
ln -s /data/tiny-imagenet/val_images_no_split/ data/tiny-imagenet200/val/images
ln -s /data/tiny-imagenet/val_images_no_split/val_annotations.txt ./data/tiny-imagenet200/val/
ln -s /data/tiny-imagenet/wnids.txt data/tiny-imagenet200/
ln -s /data/tiny-imagenet/words.txt data/tiny-imagenet200/
# optional
#  ln -s /data/tiny-imagenet/val/val_annotations.txt /data/tiny-imagenet/val_link/
